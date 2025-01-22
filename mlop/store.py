import logging
import queue
import sqlite3
import time
import threading

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Store"


class DataStore:
    def __init__(self, settings) -> None:
        self.table_data = settings.store_table_data
        self.table_file = settings.store_table_file
        self.db = f"{settings.work_dir()}/{settings.store_db}"
        self.max_size = settings.store_max_size
        self.aggregate_interval = settings.store_aggregate_interval
        self._wait = settings.store_aggregate_interval

        self.conn = sqlite3.connect(
            self.db, check_same_thread=False
        )  # isolation_level=None
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.cursor = self.conn.cursor()

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        self._thread = None
        self.start()

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_data}(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time INTEGER NOT NULL,
                step INTEGER NOT NULL,
                key TEXT NOT NULL,
                value REAL NOT NULL
            );
        """)
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_file}(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time INTEGER NOT NULL,
                step INTEGER NOT NULL,
                name TEXT NOT NULL,
                aid REAL NOT NULL
            );
        """)
        self.conn.commit()

    def insert(self, data=None, file=None, timestamp=None, step=None):
        self._queue.put((data, file, timestamp, step))

    def stop(self):
        while not self._queue.empty():
            pass
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=None)  # investigate
            self._thread = None
        self.conn.commit()
        self.conn.close()
        logger.info(f"{tag}: find saved database at {self.db}")

    def _worker(self):
        while not self._stop_event.is_set():
            batch_data, batch_file = [], []
            start = time.time()
            while (
                time.time() - start < self.aggregate_interval
                and len(batch_data) < self.max_size
                and len(batch_file) < self.max_size
            ):
                try:
                    d, f, t, s = self._queue.get(
                        timeout=max(0, self.aggregate_interval - (time.time() - start))
                    )
                    if d != {}:
                        batch_data.append(
                            {
                                "t": t,
                                "s": s,
                                "d": d,
                            }
                        )
                    if f != {}:  # file
                        batch_file.append(
                            {
                                "t": t,
                                "s": s,
                                "f": f,
                            }
                        )
                except queue.Empty:
                    continue
            self._insert(batch_data, batch_file)

    def _insert(self, d, f):
        with self._lock:
            self.conn.execute("BEGIN")
            try:
                if d != []:
                    self.cursor.executemany(
                        f"""
                        INSERT INTO {self.table_data} (time, step, key, value) VALUES (?, ?, ?, ?)
                        """,
                        [
                            (e["t"], e["s"], k, v)
                            for e in d
                            for k, v in e["d"].items()
                        ],
                    )
                    logger.info(f"{tag}: inserted {len(d)} item(s)")
                if f != []:
                    self.cursor.executemany(
                        f"""
                        INSERT INTO {self.table_file} (time, step, name, aid) VALUES (?, ?, ?, ?)
                        """,
                        [
                            (e["t"], e["s"], f"{fe._name}{fe._ext}", fe._id)
                            for e in f
                            for fe in e["f"].values()
                        ],
                    )
                    logger.info(f"{tag}: inserted {len(f)} file(s)")
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                logger.error("%s: failed to insert batch: %s", tag, e)
