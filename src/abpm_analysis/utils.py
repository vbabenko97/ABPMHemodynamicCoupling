"""
Utility Functions
==================

Logging setup, file I/O helpers, and progress tracking.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for the pipeline.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dict_to_file(data: Dict[str, Any], filepath: Path, mode: str = 'w') -> None:
    """
    Save dictionary contents to text file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        mode: File mode ('w' for write, 'a' for append)
    """
    with open(filepath, mode) as f:
        for key, value in data.items():
            f.write(f"{key}: {value}\\n")


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, name: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            name: Name of operation
        """
        self.total = total
        self.name = name
        self.current = 0
    
    def update(self, n: int = 1) -> None:
        """
        Update progress.
        
        Args:
            n: Number of items processed
        """
        self.current += n
        if self.current % max(1, self.total // 10) == 0 or self.current == self.total:
            pct = 100 * self.current / self.total
            print(f"{self.name}: {self.current}/{self.total} ({pct:.0f}%)")
    
    def finish(self) -> None:
        """Mark as complete."""
        self.current = self.total
        print(f"{self.name}: Complete ({self.total} items)")
