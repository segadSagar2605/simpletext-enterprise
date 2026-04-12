import asyncio
import json
from threading import Lock
from datetime import datetime
from typing import Set, Callable
from .logger import log_event, PerformanceTimer
import time

_broadcast_callbacks: Set[Callable] = set()
_broadcast_lock = Lock()
_event_queue: asyncio.Queue = None


def _ensure_event_queue():
    global _event_queue
    if _event_queue is None:
        try:
            _event_queue = asyncio.Queue()
        except RuntimeError:
            return None
    return _event_queue


def register_broadcast_listener(callback: Callable):
    with _broadcast_lock:
        _broadcast_callbacks.add(callback)


def unregister_broadcast_listener(callback: Callable):
    with _broadcast_lock:
        _broadcast_callbacks.discard(callback)


def _build_event_data(doc_id, event, value, unit, doc_title, meta):
    """Single place that constructs the WebSocket payload — aligned with new schema."""
    return {
        "timestamp": datetime.now().isoformat(),
        "doc_id":    doc_id,
        "doc_title": doc_title,
        "event":     event,
        "value":     f"{value:.3f}" if isinstance(value, float) and value > 0 else (value or None),
        "unit":      unit or None,
        "meta":      meta or None,
    }


async def broadcast_event(
    doc_id: int,
    event: str,
    value: float = 0,
    unit: str = "ms",
    doc_title: str = None,
    meta: dict = None,
):
    """Async broadcast — call from async endpoints or WebSocket handlers."""
    log_event(doc_id, event, doc_title=doc_title, value=value, unit=unit, meta=meta)

    event_data = _build_event_data(doc_id, event, value, unit, doc_title, meta)

    with _broadcast_lock:
        callbacks = list(_broadcast_callbacks)

    for callback in callbacks:
        try:
            await callback(event_data)
        except Exception as e:
            print(f"Broadcast callback error: {e}")


def broadcast_event_sync(
    doc_id: int,
    event: str,
    value: float = 0,
    unit: str = "ms",
    doc_title: str = None,
    meta: dict = None,
):
    """
    Sync broadcast — safe to call from background tasks and thread pools.
    Logs to CSV immediately, then queues for async WebSocket delivery.
    """
    log_event(doc_id, event, doc_title=doc_title, value=value, unit=unit, meta=meta)

    event_data = _build_event_data(doc_id, event, value, unit, doc_title, meta)

    queue = _ensure_event_queue()
    if queue:
        try:
            queue.put_nowait(event_data)
        except (asyncio.QueueFull, Exception):
            pass


async def event_broadcaster_task():
    """Background task — drains the queue and fans out to all WebSocket callbacks."""
    queue = _ensure_event_queue()
    if not queue:
        return

    while True:
        try:
            event_data = await asyncio.wait_for(queue.get(), timeout=1.0)
            with _broadcast_lock:
                callbacks = list(_broadcast_callbacks)
            for callback in callbacks:
                try:
                    await callback(event_data)
                except Exception as e:
                    print(f"Broadcast callback error: {e}")
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            print(f"Event broadcaster error: {e}")
            await asyncio.sleep(0.1)


class PerformanceTimerWithBroadcast:
    """Async context manager — times a block and broadcasts Start + Finish."""

    def __init__(self, doc_id: int, event_start: str, event_end: str,
                 doc_title: str = None, meta: dict = None):
        self.doc_id      = doc_id
        self.event_start = event_start
        self.event_end   = event_end
        self.doc_title   = doc_title
        self.meta        = dict(meta or {})
        self.start_time  = None

    async def __aenter__(self):
        await broadcast_event(self.doc_id, self.event_start,
                              value=0, unit="ms",
                              doc_title=self.doc_title,
                              meta=self.meta or None)
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self.start_time) * 1000
        await broadcast_event(self.doc_id, self.event_end,
                              value=elapsed, unit="ms",
                              doc_title=self.doc_title,
                              meta=self.meta or None)
        return False

    def update_meta(self, **kwargs):
        self.meta.update(kwargs)


class PerformanceTimerWithBroadcastSync:
    """Sync context manager — times a block and broadcasts Start + Finish."""

    def __init__(self, doc_id: int, event_start: str, event_end: str,
                 doc_title: str = None, meta: dict = None):
        self.doc_id      = doc_id
        self.event_start = event_start
        self.event_end   = event_end
        self.doc_title   = doc_title
        self.meta        = dict(meta or {})
        self.start_time  = None

    def __enter__(self):
        broadcast_event_sync(self.doc_id, self.event_start,
                             value=0, unit="ms",
                             doc_title=self.doc_title,
                             meta=self.meta or None)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self.start_time) * 1000
        broadcast_event_sync(self.doc_id, self.event_end,
                             value=elapsed, unit="ms",
                             doc_title=self.doc_title,
                             meta=self.meta or None)
        return False

    def update_meta(self, **kwargs):
        self.meta.update(kwargs)


class SimplePerformanceFormatter:
    """Formats a WebSocket event payload into a human-readable string for the status bar."""

    @staticmethod
    def format_for_display(event_data: dict) -> str:
        doc_id = event_data.get("doc_id")
        event  = event_data.get("event", "")
        value  = event_data.get("value")
        unit   = event_data.get("unit", "ms")

        base = f"[Doc {doc_id}] {event}" if doc_id else event

        if value and str(value) not in ("0", "0.000", "None"):
            return f"{base} ({value}{unit})"
        return base
