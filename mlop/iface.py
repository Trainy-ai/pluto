import json
import logging
import queue
import threading
import time
import _io

import httpx

from .file import File
from .sets import Settings

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Interface"


class ServerInterface:
    def __init__(self, settings: Settings) -> None:
        self.url_data = settings.url_data
        self.url_file = settings.url_file
        self.url_view_op = (
            f"{settings.url_view}/{settings.user}/{settings.project}/{settings._op_id}"
        )

        self.headers = {
            "Authorization": f"Bearer {settings.auth}",
            "Content-Type": "application/json",
            "User-Agent": f"{__name__.split('.')[0]}",
            "X-Run-Id": f"{settings._op_id}",
            "X-Run-Name": f"{settings._op_name}",
            "X-Project-Name": f"{settings.project}",
        }
        self.headers_data = self.headers.copy()
        self.headers_data.update({"Content-Type": "application/x-ndjson"})

        self.client = httpx.Client(
            http2=True,
            verify=False,  # fix
            proxy=settings.http_proxy or settings.https_proxy or None,
            limits=httpx.Limits(
                max_keepalive_connections=settings.x_file_stream_max_conn,
                max_connections=settings.x_file_stream_max_conn,
            ),
            timeout=httpx.Timeout(
                settings.x_file_stream_timeout_seconds,
                # connect=settings.x_file_stream_timeout_seconds,
            ),
        )

        self.max_size = settings.x_file_stream_max_size
        self.retry_max = settings.x_file_stream_retry_max
        self.retry_wait_min = settings.x_file_stream_retry_wait_min_seconds
        self.retry_wait_max = settings.x_file_stream_retry_wait_max_seconds
        self.transmit_interval = settings.x_file_stream_transmit_interval

        self._wait = settings.x_internal_check_process
        self._stop_event = threading.Event()
        self._queue_data = queue.Queue()
        self._buffer_data = None
        self._thread_data = None

        self._queue_storage = queue.Queue()
        self._buffer_storage = None
        self._queue_file = queue.Queue()

    def start(self) -> None:
        if self._thread_data is None:
            self._thread_data = threading.Thread(
                target=self._worker_publish,
                args=(
                    self.url_data,
                    self.headers_data,
                    self._queue_data,
                    self._buffer_data,
                    self._stop_event.is_set,
                ),
                daemon=True,
            )
            self._thread_data.start()

    def publish(
        self,
        data: dict[str, any] | None = None,
        file: None = None,
        timestamp: int | None = None,
        step: int | None = None,
    ) -> None:
        if data:
            self._queue_data.put(
                make_compat_data_v1(data, timestamp, step), block=False
            )
        if file:
            self._queue_storage.put(
                make_compat_file_v1(file, timestamp, step), block=False
            )
            r = self._post_v1(
                self.url_file,
                self.headers,
                self._queue_storage,
                self._buffer_storage,
                client=self.client,
            )
            try:
                for f in file.values():
                    for e in r.json():
                        if e["name"] == f"{f._name}{f._ext}":
                            f._url = e["url"]
                            self._queue_file.put(open(f._path, "rb"), block=False)
                            # self._queue_file.put(f, block=False)
                            s = self._put_v1(
                                f._url,
                                {
                                    "Content-Type": f._type,  # "application/octet-stream"
                                },
                                self._queue_file,
                                client=self.client,
                            )
                            assert s.status_code in [200, 201]
            except Exception as e:
                logger.critical("%s: failed to send files: %s", tag, e)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread_data is not None:
            self._thread_data.join(timeout=self._wait)
            self._thread_data = None
        logger.info(f"{tag}: find uploaded data at {self.url_view_op}")

    def _worker_publish(self, e, h, q, b, stop):
        while not q.empty() or not stop():
            if not q.empty():
                _ = self._post_v1(
                    e,
                    h,
                    q,
                    b,
                    client=self.client,
                )

    def _queue_iter(self, q, b):
        s = time.time()
        while (
            len(b) < self.max_size
            and (time.time() - s) < self.transmit_interval
        ):
            try:
                v = q.get(block=False)
                b.append(v)
                yield v
            except queue.Empty:
                break

    def _put_v1(self, url, headers, q, client=None):
        try:
            r = client.put(
                url,
                content=q.get(),  # fix: add retry
                headers=headers,
            )
            if r.status_code in [200, 201]:
                return r
            else:
                logger.error(
                    f"{tag}: server responded error {r.status_code} during PUT: {r.text}"
                )
        except Exception as e:
            logger.error("%s: no response received: %s", tag, e)

    def _post_v1(self, url, headers, q, b, client=None, retry=0):
        b = []
        try:
            s = time.time()
            r = client.post(
                url,
                content=self._queue_iter(q, b),  # iter(q.get, None),
                headers=headers,
            )
            if r.status_code in [200, 201]:
                logger.info(
                    f"{tag}: sent {len(b)} item(s) at {len(b) / (time.time() - s):.2f} items/s"
                )
                return r
            else:
                logger.error(
                    f"{tag}: server responded error {r.status_code} for {len(b)} item(s) during POST: {r.text}"
                )
        except Exception as e:
            logger.error("%s: no response received during POST: %s", tag, e)

        retry += 1
        if retry < self.retry_max:
            logger.warning(
                f"{tag}: retry {retry}/{self.retry_max} for {len(b)} item(s)"
            )
            time.sleep(min(self.retry_wait_min * (2**retry), self.retry_wait_max))
            for i in b:
                q.put(i, block=False)
            return self._post_v1(url, headers, q, b, client=client, retry=retry)
        else:
            logger.critical(f"{tag}: failed to send {len(b)} item(s)")
            return None


def make_compat_data_v1(data, timestamp, step):
    batch = []
    i = {
        "time": int(timestamp * 1000),  # convert to ms
        "step": int(step),
        "data": data,
    }
    batch.append(json.dumps(i))
    return ("\n".join(batch) + "\n").encode("utf-8")


def make_compat_file_v1(file, timestamp, step):
    batch = []
    for k, v in file.items():
        i = {
            "fileName": f"{v._name}{v._ext}",
            "size": v._size,
            "fileType": v._ext[1:],
            "logName": k,
            "step": step,
        }
        batch.append(i)
    return json.dumps({"files": batch}).encode()
