"""
Utility Functions
==================

Progress tracking for long-running operations.
"""


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
        if self.total <= 0:
            return
        if self.current % max(1, self.total // 10) == 0 or self.current == self.total:
            pct = 100 * self.current / self.total
            print(f"{self.name}: {self.current}/{self.total} ({pct:.0f}%)")

    def finish(self) -> None:
        """Mark as complete."""
        self.current = self.total
        print(f"{self.name}: Complete ({self.total} items)")
