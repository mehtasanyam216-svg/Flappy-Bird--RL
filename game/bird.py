"""
bird.py — Bird entity with gravity-based physics and flap mechanic.
"""

GRAVITY = 0.5        
FLAP_STRENGTH = -8.0 
MAX_FALL_SPEED = 10.0
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
BIRD_START_X = 80    
BIRD_START_Y = 256   

class Bird:
    """Represents the player-controlled bird."""

    def __init__(self):
        self.x = BIRD_START_X
        self.y = BIRD_START_Y
        self.velocity = 0.0
        self.width = BIRD_WIDTH
        self.height = BIRD_HEIGHT

    def flap(self):
        """Apply an upward impulse."""
        self.velocity = FLAP_STRENGTH

    def update(self):
        """Advance physics by one frame: apply gravity, clamp speed, move."""
        self.velocity += GRAVITY
        if self.velocity > MAX_FALL_SPEED:
            self.velocity = MAX_FALL_SPEED
        self.y += self.velocity

    def get_rect(self):
        """Return (x, y, width, height) bounding box for collision checks."""
        return (self.x, self.y, self.width, self.height)

    def reset(self):
        """Restore the bird to its starting state."""
        self.x = BIRD_START_X
        self.y = BIRD_START_Y
        self.velocity = 0.0
