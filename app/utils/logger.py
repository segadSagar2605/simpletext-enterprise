import csv
import os
import time
from threading import Lock
from datetime import datetime
from pathlib import Path

# Thread-safe lock for CSV writes
_csv_lock = Lock()

# CSV file path - stores in project root
CSV_FILE = "granular_performance.csv"


def _ensure_csv_header():
    """Create CSV with headers if it doesn't exist."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'doc_id', 'doc_title', 'chunk_id', 'event', 'duration_ms'])
            writer.writeheader()


def log_event(doc_id, event: str, duration_ms: float, doc_title: str = None, chunk_id: str = None):
    """
    Thread-safe performance logging function.
    
    Args:
        doc_id: Document ID (or None for system-level events)
        event: Event name (e.g., "Upload Start", "Embedding Finish")
        duration_ms: Duration in milliseconds (0 for events with no duration)
        doc_title: Optional document title for context
        chunk_id: Optional chunk identifier (e.g., "p001_c002")
    """
    with _csv_lock:
        _ensure_csv_header()
        
        timestamp = datetime.now().isoformat()
        
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'doc_id', 'doc_title', 'chunk_id', 'event', 'duration_ms'])
            writer.writerow({
                'timestamp': timestamp,
                'doc_id': doc_id,
                'doc_title': doc_title or '',
                'chunk_id': chunk_id or '',
                'event': event,
                'duration_ms': f"{duration_ms:.3f}"
            })


class PerformanceTimer:
    """Context manager for measuring execution time with automatic logging."""
    
    def __init__(self, doc_id, event_start: str, event_end: str):
        """
        Args:
            doc_id: Document ID
            event_start: Name of the start event (logged with 0 duration)
            event_end: Name of the end event (logged with elapsed duration)
        """
        self.doc_id = doc_id
        self.event_start = event_start
        self.event_end = event_end
        self.start_time = None
    
    def __enter__(self):
        log_event(self.doc_id, self.event_start, 0)
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self.start_time) * 1000  # Convert to ms
        log_event(self.doc_id, self.event_end, elapsed)
        return False
