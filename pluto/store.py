import logging
import queue
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .sets import Settings

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = 'Store'


class DataStore:
    def __init__(self, config: dict, settings: Settings) -> None:
        self.settings = settings

        self.db = f'{settings.get_dir()}/{settings.store_db}'

        self.conn = sqlite3.connect(
            self.db, check_same_thread=False
        )  # isolation_level=None
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self.cursor = self.conn.cursor()

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._queue: queue.Queue[
            Tuple[
                Optional[Dict[str, Any]],
                Optional[Dict[str, Any]],
                Optional[Dict[str, Any]],
                Optional[float],
                Optional[int],
            ]
        ] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self.start()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.settings.store_table_num}(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time INTEGER NOT NULL,
                step INTEGER NOT NULL,
                key TEXT NOT NULL,
                value REAL NOT NULL
            );
        """)
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.settings.store_table_file}(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time INTEGER NOT NULL,
                step INTEGER NOT NULL,
                name TEXT NOT NULL,
                aid REAL NOT NULL
            );
        """)
        self.conn.commit()

    def insert(
        self,
        num: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        file: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        step: Optional[int] = None,
    ) -> None:
        self._queue.put((num, data, file, timestamp, step))

    def stop(self) -> None:
        while not self._queue.empty():
            pass
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=None)  # TODO: investigate hanging
            self._thread = None
        self.conn.commit()
        self.conn.close()
        logger.info(f'{tag}: find saved database at {self.db}')

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            batch_num: List[Dict[str, Any]] = []
            batch_file: List[Dict[str, Any]] = []
            start = time.time()
            while (
                time.time() - start < self.settings.store_aggregate_interval
                and len(batch_num) < self.settings.store_max_size
                and len(batch_file) < self.settings.store_max_size
            ):
                try:  # TODO: add data support
                    n, d, f, t, s = self._queue.get(
                        timeout=max(
                            0,
                            self.settings.store_aggregate_interval
                            - (time.time() - start),
                        )
                    )
                    if n != {}:
                        batch_num.append(
                            {
                                't': t,
                                's': s,
                                'n': n,
                            }
                        )
                    if f != {}:
                        batch_file.append(
                            {
                                't': t,
                                's': s,
                                'f': f,
                            }
                        )
                except queue.Empty:
                    continue
            self._insert(batch_num, batch_file)

    def _insert(self, d: List[Dict[str, Any]], f: List[Dict[str, Any]]) -> None:
        with self._lock:
            self.conn.execute('BEGIN')
            try:
                if d != []:
                    self.cursor.executemany(
                        f"""
                        INSERT INTO {self.settings.store_table_num}
                        (time, step, key, value)
                        VALUES (?, ?, ?, ?)
                        """,
                        [(e['t'], e['s'], k, v) for e in d for k, v in e['n'].items()],
                    )
                    logger.info(f'{tag}: inserted {len(d)} line(s)')
                if f != []:
                    self.cursor.executemany(
                        f"""
                        INSERT INTO {self.settings.store_table_file}
                        (time, step, name, aid)
                        VALUES (?, ?, ?, ?)
                        """,
                        [
                            (e['t'], e['s'], f'{fe._name}{fe._ext}', fe._id)
                            for e in f
                            # for fe in e["f"].values()
                            for fel in e['f'].values()
                            for fe in fel
                        ],
                    )
                    logger.info(f'{tag}: inserted {len(f)} file(s)')
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                logger.error('%s: failed to insert batch: %s', tag, e)
