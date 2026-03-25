"""
pipe.py — Pipe obstacle with random gap positioning and horizontal movement.
"""
import random

# ─── Constants ───────────────────────────────────────────────────────
PIPE_WIDTH = 52
PIPE_GAP = 150          # Vertical gap between top and bottom pipe
PIPE_SPEED = 3          # Pixels the pipe moves left per frame
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 512
PIPE_MIN_TOP = 50       # Minimum height for the top pipe
PIPE_MAX_TOP = SCREEN_HEIGHT - PIPE_GAP - 50  # Maximum height for the top pipe


class Pipe:
    """A single pair of top/bottom pipes."""

    def __init__(self, x: float = SCREEN_WIDTH):
        self.x = x
        self.width = PIPE_WIDTH
        # Random height for the top pipe — the gap starts right below it
        self.top_height = random.randint(PIPE_MIN_TOP, PIPE_MAX_TOP)
        self.bottom_y = self.top_height + PIPE_GAP
        self.passed = False  # True once the bird has cleared this pipe pair

    # ── Derived properties ───────────────────────────────────────────
    @property
    def gap_center_y(self) -> float:
        """Vertical centre of the gap (useful for state representation)."""
        return self.top_height + PIPE_GAP / 2

    # ── Movement ─────────────────────────────────────────────────────
    def update(self):
        """Move the pipe to the left by PIPE_SPEED pixels."""
        self.x -= PIPE_SPEED

    def is_offscreen(self) -> bool:
        """Return True when the pipe has fully scrolled past the left edge."""
        return self.x + self.width < 0

    # ── Collision helpers ────────────────────────────────────────────
    def get_top_rect(self):
        """Bounding box for the top pipe: (x, y, width, height)."""
        return (self.x, 0, self.width, self.top_height)

    def get_bottom_rect(self):
        """Bounding box for the bottom pipe: (x, y, width, height)."""
        bottom_height = SCREEN_HEIGHT - self.bottom_y
        return (self.x, self.bottom_y, self.width, bottom_height)
