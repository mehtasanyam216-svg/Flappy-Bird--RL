"""
flappy_env.py — RL-style environment wrapper for Flappy Bird.

Exposes:
    reset()            → initial state vector
    step(action)       → (next_state, reward, done)
    render()           → optional pygame visualisation
    close()            → clean up pygame resources
"""
import pygame
from game.bird import Bird, BIRD_START_X, BIRD_WIDTH
from game.pipe import Pipe, PIPE_WIDTH, PIPE_SPEED, PIPE_GAP, SCREEN_WIDTH, SCREEN_HEIGHT

# ─── Environment Constants ───────────────────────────────────────────
FPS = 60
PIPE_SPAWN_INTERVAL = 90     # Frames between spawning a new pipe pair
GROUND_HEIGHT = SCREEN_HEIGHT  # The ground is at the bottom of the screen

# Reward shaping
REWARD_ALIVE = 0.0        # Removed — don't reward mere survival
REWARD_PASS_PIPE = 50.0   # Stronger signal for clearing a pipe
REWARD_DEATH = -100.0

# Colours (R, G, B)
COLOR_SKY = (135, 206, 235)
COLOR_BIRD = (255, 255, 0)
COLOR_PIPE = (0, 200, 0)
COLOR_GROUND = (222, 184, 135)
COLOR_TEXT = (255, 255, 255)


class FlappyEnv:
    """
    Flappy Bird environment with an RL-compatible interface.

    Actions:
        0 → do nothing
        1 → flap
    """

    def __init__(self, render_mode: bool = False):
        """
        Args:
            render_mode: If True, initialise pygame display for visualisation.
        """
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird — RL Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 24)

        # Game objects
        self.bird = Bird()
        self.pipes: list[Pipe] = []
        self.score = 0
        self.frame_count = 0

    # ══════════════════════════════════════════════════════════════════
    #  RL Interface
    # ══════════════════════════════════════════════════════════════════

    def reset(self) -> list[float]:
        """Reset the environment and return the initial state vector."""
        self.bird.reset()
        self.pipes = [Pipe()]
        self.score = 0
        self.frame_count = 0
        return self._get_state()

    def step(self, action: int) -> tuple[list[float], float, bool]:
        """
        Execute one environment step.

        Args:
            action: 0 (do nothing) or 1 (flap).

        Returns:
            (state, reward, done)
        """
        # ── Apply action ─────────────────────────────────────────────
        if action == 1:
            self.bird.flap()

        # ── Update bird physics ──────────────────────────────────────
        self.bird.update()

        # ── Update pipes ─────────────────────────────────────────────
        self.frame_count += 1
        if self.frame_count % PIPE_SPAWN_INTERVAL == 0:
            self.pipes.append(Pipe())

        for pipe in self.pipes:
            pipe.update()

        # Remove off-screen pipes to save memory
        self.pipes = [p for p in self.pipes if not p.is_offscreen()]

        # ── Check for pipe passing (score) ───────────────────────────
        reward = REWARD_ALIVE
        for pipe in self.pipes:
            # Bird's right edge has just crossed the pipe's right edge
            if (not pipe.passed
                    and pipe.x + pipe.width < self.bird.x):
                pipe.passed = True
                self.score += 1
                reward = REWARD_PASS_PIPE

        # ── Collision detection ──────────────────────────────────────
        done = self._check_collision()
        if done:
            reward = REWARD_DEATH

        state = self._get_state()
        return state, reward, done

    # ══════════════════════════════════════════════════════════════════
    #  Rendering
    # ══════════════════════════════════════════════════════════════════

    def render(self):
        """Draw the current frame (only if render_mode is True)."""
        if not self.render_mode:
            return

        self.screen.fill(COLOR_SKY)

        # Draw pipes
        for pipe in self.pipes:
            # Top pipe
            tx, ty, tw, th = pipe.get_top_rect()
            pygame.draw.rect(self.screen, COLOR_PIPE, (tx, ty, tw, th))
            # Bottom pipe
            bx, by, bw, bh = pipe.get_bottom_rect()
            pygame.draw.rect(self.screen, COLOR_PIPE, (bx, by, bw, bh))

        # Draw bird
        bx, by, bw, bh = self.bird.get_rect()
        pygame.draw.rect(self.screen, COLOR_BIRD, (bx, by, bw, bh))

        # Score overlay
        score_surf = self.font.render(f"Score: {self.score}", True, COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        """Clean up pygame resources."""
        if self.render_mode:
            pygame.quit()

    # ══════════════════════════════════════════════════════════════════
    #  Internal helpers
    # ══════════════════════════════════════════════════════════════════

    def _get_state(self) -> list[float]:
        """
        Build a 4-element numerical state vector:
            [bird_y, bird_velocity, horiz_dist_to_next_pipe, vert_dist_to_gap_center]

        If no pipes are ahead of the bird, default distances are used so
        the vector length stays fixed.
        """
        next_pipe = self._get_next_pipe()
        if next_pipe is not None:
            horiz_dist = next_pipe.x - self.bird.x
            vert_dist = next_pipe.gap_center_y - self.bird.y
        else:
            # Fallback: no pipe on screen yet (shouldn't happen after reset)
            horiz_dist = SCREEN_WIDTH
            vert_dist = 0.0

        return [
            self.bird.y,
            self.bird.velocity,
            horiz_dist,
            vert_dist,
        ]

    def _get_next_pipe(self) -> Pipe | None:
        """Return the nearest pipe whose right edge is still ahead of the bird."""
        for pipe in sorted(self.pipes, key=lambda p: p.x):
            if pipe.x + pipe.width > self.bird.x:
                return pipe
        return None

    def _check_collision(self) -> bool:
        """Return True if the bird has collided with a pipe or the ground/ceiling."""
        bx, by, bw, bh = self.bird.get_rect()

        # Ground or ceiling
        if by + bh >= SCREEN_HEIGHT or by <= 0:
            return True

        # Pipe rectangles (axis-aligned bounding-box test)
        for pipe in self.pipes:
            if self._rects_overlap((bx, by, bw, bh), pipe.get_top_rect()):
                return True
            if self._rects_overlap((bx, by, bw, bh), pipe.get_bottom_rect()):
                return True

        return False

    @staticmethod
    def _rects_overlap(a, b) -> bool:
        """AABB overlap test. Each rect is (x, y, width, height)."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return (ax < bx + bw and ax + aw > bx and
                ay < by + bh and ay + ah > by)
