import asyncio
import json
from threading import Lock
from datetime import datetime
from typing import Set, Callable, Any
from .logger import log_event, PerformanceTimer
import time

# Thread-safe set of async callbacks for event broadcasting
_broadcast_callbacks: Set[Callable] = set()
_broadcast_lock = Lock()

# Global queue for async event broadcasting (thread-safe)
_event_queue: asyncio.Queue = None


def _ensure_event_queue():
    """Lazily initialize the event queue."""
    global _event_queue
    if _event_queue is None:
        try:
            _event_queue = asyncio.Queue()
        except RuntimeError:
            # If no event loop running, create one
            return None
    return _event_queue


def register_broadcast_listener(callback: Callable):
    """
    Register a callback to receive broadcast events.
    The callback should be an async function that accepts an event dict.
    
    Args:
        callback: Async function(event_dict) to be called on each broadcast
    """
    with _broadcast_lock:
        _broadcast_callbacks.add(callback)


def unregister_broadcast_listener(callback: Callable):
    """Unregister a broadcast listener."""
    with _broadcast_lock:
        _broadcast_callbacks.discard(callback)


async def broadcast_event(doc_id: int, event: str, duration_ms: float = 0, doc_title: str = None, chunk_id: str = None):
    """
    Async broadcast a performance event to all registered listeners.
    Call this from async context (e.g., endpoints, WebSocket handlers).
    
    Args:
        doc_id: Document ID
        event: Event name
        duration_ms: Duration in milliseconds (0 for span start events)
        doc_title: Optional document title
        chunk_id: Optional chunk identifier
    """
    # Log to CSV (thread-safe)
    log_event(doc_id, event, duration_ms, doc_title, chunk_id)
    
    # Broadcast to WebSocket listeners
    event_data = {
        "timestamp": datetime.now().isoformat(),
        "doc_id": doc_id,
        "doc_title": doc_title,
        "chunk_id": chunk_id,
        "event": event,
        "duration_ms": f"{duration_ms:.3f}" if duration_ms > 0 else None
    }
    
    # Call all registered callbacks
    with _broadcast_lock:
        callbacks = list(_broadcast_callbacks)
    
    for callback in callbacks:
        try:
            await callback(event_data)
        except Exception as e:
            print(f"Broadcast callback error: {e}")


def broadcast_event_sync(doc_id: int, event: str, duration_ms: float = 0, doc_title: str = None, chunk_id: str = None):
    """
    Synchronous version of broadcast_event for use in background tasks.
    Logs to CSV and queues event for async broadcasting.
    
    This is safe to call from sync contexts (background tasks, thread pool, etc).
    
    Args:
        doc_id: Document ID
        event: Event name
        duration_ms: Duration in milliseconds (0 for span start events)
        doc_title: Optional document title
        chunk_id: Optional chunk identifier
    """
    # Log to CSV (thread-safe)
    log_event(doc_id, event, duration_ms, doc_title, chunk_id)
    
    # Prepare event data
    event_data = {
        "timestamp": datetime.now().isoformat(),
        "doc_id": doc_id,
        "doc_title": doc_title,
        "chunk_id": chunk_id,
        "event": event,
        "duration_ms": f"{duration_ms:.3f}" if duration_ms > 0 else None
    }
    
    # Try to queue for async broadcasting (non-blocking, safe to fail)
    queue = _ensure_event_queue()
    if queue:
        try:
            queue.put_nowait(event_data)
        except asyncio.QueueFull:
            pass  # Queue full, event will be missed but won't crash
        except Exception:
            pass  # Other errors, silently ignore


async def event_broadcaster_task():
    """
    Background task that consumes events from the queue and broadcasts them.
    Run this as a background task in the app.
    """
    queue = _ensure_event_queue()
    if not queue:
        return
    
    while True:
        try:
            event_data = await asyncio.wait_for(queue.get(), timeout=1.0)
            # Call registered callbacks
            with _broadcast_lock:
                callbacks = list(_broadcast_callbacks)
            
            for callback in callbacks:
                try:
                    await callback(event_data)
                except Exception as e:
                    print(f"Broadcast callback error: {e}")
        except asyncio.TimeoutError:
            # Timeout is normal, just continue waiting
            continue
        except Exception as e:
            print(f"Event broadcaster error: {e}")
            await asyncio.sleep(0.1)  # Brief delay before retry


class PerformanceTimerWithBroadcast:
    """
    Context manager for measuring execution time with automatic logging and broadcasting.
    Extends PerformanceTimer to also broadcast events to WebSocket listeners.
    """
    
    def __init__(self, doc_id: int, event_start: str, event_end: str):
        """
        Args:
            doc_id: Document ID
            event_start: Name of the start event
            event_end: Name of the end event
        """
        self.doc_id = doc_id
        self.event_start = event_start
        self.event_end = event_end
        self.start_time = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await broadcast_event(self.doc_id, self.event_start, 0)
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        elapsed = (time.perf_counter() - self.start_time) * 1000
        await broadcast_event(self.doc_id, self.event_end, elapsed)
        return False


class PerformanceTimerWithBroadcastSync:
    """
    Context manager for synchronous code (e.g., background tasks).
    Logs and queues events for async broadcasting.
    Use this in background_tasks or thread contexts.
    """
    
    def __init__(self, doc_id: int, event_start: str, event_end: str):
        """
        Args:
            doc_id: Document ID
            event_start: Name of the start event
            event_end: Name of the end event
        """
        self.doc_id = doc_id
        self.event_start = event_start
        self.event_end = event_end
        self.start_time = None
    
    def __enter__(self):
        """Synchronous context manager entry."""
        broadcast_event_sync(self.doc_id, self.event_start, 0)
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Synchronous context manager exit."""
        elapsed = (time.perf_counter() - self.start_time) * 1000
        broadcast_event_sync(self.doc_id, self.event_end, elapsed)
        return False


class SimplePerformanceFormatter:
    """Formats performance event data for display in UI."""
    
    @staticmethod
    def format_for_display(event_data: dict) -> str:
        """
        Convert event data to user-friendly display string.
        
        Args:
            event_data: Dict with timestamp, doc_id, event, duration_ms
            
        Returns:
            Formatted string like "[Doc 42] Embedding Finish (123.456ms)"
        """
        doc_id = event_data.get("doc_id")
        event = event_data.get("event", "")
        duration_ms = event_data.get("duration_ms")
        
        if doc_id:
            base = f"[Doc {doc_id}] {event}"
        else:
            base = event
        
        if duration_ms and duration_ms != "None":
            return f"{base} ({duration_ms}ms)"
        
        return base
