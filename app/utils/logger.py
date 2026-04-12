import csv
import json
import os
import time
from threading import Lock
from datetime import datetime

_csv_lock = Lock()
CSV_FILE = "granular_performance.csv"
FIELDNAMES = ['timestamp', 'doc_id', 'doc_title', 'event', 'value', 'unit', 'meta']


def _ensure_csv_header():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def log_event(
    doc_id,
    event: str,
    duration_ms: float = 0,
    doc_title: str = None,
    chunk_id: str = None,
    value: float = None,
    unit: str = None,
    meta: dict = None,
):
    """
    Thread-safe performance event logger.

    Schema:
        value  – numeric measurement (duration, count, throughput, etc.)
        unit   – "ms", "count", "chunks", "batches", "chunks/s", etc.
        meta   – dict serialised to JSON; carries structured per-event context:
                 {"parent_blocks": 12, "total_child_chunks": 47, "batch": 1, ...}

    Backward compatible: callers passing only duration_ms still work.
    """
    if value is None:
        value = duration_ms
        if unit is None and duration_ms > 0:
            unit = "ms"

    if chunk_id:
        meta = dict(meta or {})
        meta['chunk_id'] = chunk_id

    with _csv_lock:
        _ensure_csv_header()
        with open(CSV_FILE, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow({
                'timestamp': datetime.now().isoformat(),
                'doc_id':    doc_id or '',
                'doc_title': doc_title or '',
                'event':     event,
                'value':     f"{value:.3f}" if isinstance(value, float) else (value if value is not None else ''),
                'unit':      unit or '',
                'meta':      json.dumps(meta) if meta else '',
            })


class PerformanceTimer:
    """
    Sync context manager — measures elapsed time and logs Start + Finish events.

    Extra meta can be injected at any point inside the block via update_meta().

        with PerformanceTimer(doc_id, "Chunking Start", "Chunking Finish",
                              doc_title=title) as t:
            chunks = do_chunking()
            t.update_meta(parent_blocks=len(chunks))
    """

    def __init__(self, doc_id, event_start: str, event_end: str,
                 doc_title: str = None, meta: dict = None):
        self.doc_id      = doc_id
        self.event_start = event_start
        self.event_end   = event_end
        self.doc_title   = doc_title
        self.meta        = dict(meta or {})
        self.start_time  = None

    def __enter__(self):
        log_event(self.doc_id, self.event_start,
                  doc_title=self.doc_title,
                  value=0, unit='ms',
                  meta=self.meta or None)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self.start_time) * 1000
        log_event(self.doc_id, self.event_end,
                  doc_title=self.doc_title,
                  value=elapsed, unit='ms',
                  meta=self.meta or None)
        return False

    def update_meta(self, **kwargs):
        """Accumulate extra context to be written on the Finish event."""
        self.meta.update(kwargs)
