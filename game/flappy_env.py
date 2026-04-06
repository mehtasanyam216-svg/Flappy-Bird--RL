"""
flappy_env.py — RL-style environment wrapper for Flappy Bird.

Exposes:
    reset()            → initial state vector
    step(action)       → (next_state, reward, done)
    render()           → optional pygame visualisation
    close()            → clean up pygame resources
"""
import random
import pygame
from game.bird import Bird, BIRD_START_X, BIRD_WIDTH
from game.pipe import Pipe, PIPE_WIDTH, PIPE_SPEED, PIPE_GAP, SCREEN_WIDTH, SCREEN_HEIGHT

FPS = 60
PIPE_SPAWN_INTERVAL = 90   
GROUND_HEIGHT = SCREEN_HEIGHT

REWARD_ALIVE = 0.0      
REWARD_PASS_PIPE = 50.0 
REWARD_DEATH = -100.0

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
            self.font = pygame.font.SysFont("arial", 24, bold=True)
            self.clouds = [(random.randint(0, SCREEN_WIDTH), random.randint(20, 250), random.uniform(0.3, 1.0)) for _ in range(6)]
            self.ground_scroll = 0

        self.bird = Bird()
        self.pipes: list[Pipe] = []
        self.score = 0
        self.frame_count = 0

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
        if action == 1:
            self.bird.flap()

        self.bird.update()

        self.frame_count += 1
        if self.frame_count % PIPE_SPAWN_INTERVAL == 0:
            self.pipes.append(Pipe())

        for pipe in self.pipes:
            pipe.update()

        self.pipes = [p for p in self.pipes if not p.is_offscreen()]

       reward = 0.1  

next_pipe = self._get_next_pipe()
if next_pipe:
    vertical_error = abs(next_pipe.gap_center_y - self.bird.y)
    reward -= vertical_error * 0.01  


if action == 1:
    reward -= 0.05
        for pipe in self.pipes:
            if (not pipe.passed
                    and pipe.x + pipe.width < self.bird.x):
                pipe.passed = True
                self.score += 1
                reward += 10

        done = self._check_collision()
        if done:
           reward = -100

        state = self._get_state()
        return state, reward, done

    def render(self):
        """Draw the current frame (only if render_mode is True)."""
        if not self.render_mode:
            return

        # 1. Draw gradient sky
        top_color = (135, 206, 235)  # COLOR_SKY
        bottom_color = (200, 230, 255)
        for y in range(SCREEN_HEIGHT):
            blend = y / SCREEN_HEIGHT
            color = (
                int(top_color[0] + (bottom_color[0] - top_color[0]) * blend),
                int(top_color[1] + (bottom_color[1] - top_color[1]) * blend),
                int(top_color[2] + (bottom_color[2] - top_color[2]) * blend)
            )
            pygame.draw.line(self.screen, color, (0, y), (SCREEN_WIDTH, y))

        # 2. Update and draw scrolling clouds
        for i, (cx, cy, speed) in enumerate(self.clouds):
            self.clouds[i] = (cx - speed, cy, speed)
            if self.clouds[i][0] < -60:
                self.clouds[i] = (SCREEN_WIDTH + 50, random.randint(20, 250), speed)
            
            # Draw fluffy cloud out of circles
            cloud_color = (255, 255, 255)
            x = int(self.clouds[i][0])
            y = int(cy)
            pygame.draw.circle(self.screen, cloud_color, (x, y), 20)
            pygame.draw.circle(self.screen, cloud_color, (x + 20, y - 5), 25)
            pygame.draw.circle(self.screen, cloud_color, (x + 40, y), 20)
            pygame.draw.circle(self.screen, cloud_color, (x + 20, y + 10), 18)

        # 3. Draw pipes with realistic visual caps and borders
        cap_h = 24
        for pipe in self.pipes:
            # Top Pipe
            tx, ty, tw, th = pipe.get_top_rect()
            pygame.draw.rect(self.screen, COLOR_PIPE, (tx, ty, tw, th))
            pygame.draw.rect(self.screen, (0, 100, 0), (tx, ty, tw, th), 3)
            # Cap
            pygame.draw.rect(self.screen, COLOR_PIPE, (tx - 4, ty + th - cap_h, tw + 8, cap_h))
            pygame.draw.rect(self.screen, (0, 100, 0), (tx - 4, ty + th - cap_h, tw + 8, cap_h), 3)
            # Highlight
            pygame.draw.line(self.screen, (100, 255, 100), (tx + 4, ty), (tx + 4, ty + th - cap_h), 3)

            # Bottom Pipe
            bx, by, bw, bh = pipe.get_bottom_rect()
            pygame.draw.rect(self.screen, COLOR_PIPE, (bx, by, bw, bh))
            pygame.draw.rect(self.screen, (0, 100, 0), (bx, by, bw, bh), 3)
            # Cap
            pygame.draw.rect(self.screen, COLOR_PIPE, (bx - 4, by, bw + 8, cap_h))
            pygame.draw.rect(self.screen, (0, 100, 0), (bx - 4, by, bw + 8, cap_h), 3)
            # Highlight
            pygame.draw.line(self.screen, (100, 255, 100), (bx + 4, by + cap_h), (bx + 4, by + bh), 3)

        # 4. Draw moving ground / mud base
        self.ground_scroll = (self.ground_scroll - PIPE_SPEED) % 40
        ground_h = 25
        pygame.draw.rect(self.screen, (222, 216, 149), (0, SCREEN_HEIGHT - ground_h, SCREEN_WIDTH, ground_h))
        pygame.draw.rect(self.screen, (115, 191, 46), (0, SCREEN_HEIGHT - ground_h, SCREEN_WIDTH, 12)) # Grass top
        pygame.draw.line(self.screen, (84, 56, 71), (0, SCREEN_HEIGHT - ground_h), (SCREEN_WIDTH, SCREEN_HEIGHT - ground_h), 4)

        for x in range(-40, SCREEN_WIDTH + 40, 40):
            # Ground texture lines
            stripe_rect = (x + self.ground_scroll, SCREEN_HEIGHT - ground_h + 12, 18, ground_h - 12)
            pygame.draw.rect(self.screen, (211, 205, 135), stripe_rect)
            pygame.draw.line(self.screen, (84, 56, 71), (x + self.ground_scroll, SCREEN_HEIGHT - ground_h + 12), (x + self.ground_scroll - 10, SCREEN_HEIGHT), 2)

        # 5. Draw the bird with eye, wing, beak and rotation based on velocity
        bx, by, bw, bh = self.bird.get_rect()
        angle = max(-90.0, min(30.0, -self.bird.velocity * 4.0)) # Tilt down when falling, up when flying

        bird_surf = pygame.Surface((bw, bh), pygame.SRCALPHA)
        # Body
        pygame.draw.ellipse(bird_surf, COLOR_BIRD, (0, 0, bw, bh))
        pygame.draw.ellipse(bird_surf, (200, 150, 0), (0, 0, bw, bh), 2)
        # Eye
        pygame.draw.circle(bird_surf, (255, 255, 255), (bw - 10, 8), 6)
        pygame.draw.circle(bird_surf, (0, 0, 0), (bw - 8, 8), 2)
        # Beak
        pygame.draw.polygon(bird_surf, (255, 140, 0), [(bw - 5, 12), (bw + 6, 15), (bw - 3, 20)])
        pygame.draw.polygon(bird_surf, (200, 100, 0), [(bw - 5, 12), (bw + 6, 15), (bw - 3, 20)], 1)
        # Flapping Wing
        wing_y = 10 if self.bird.velocity < 0 else 14
        pygame.draw.ellipse(bird_surf, (255, 255, 255), (6, wing_y, 14, 8))
        pygame.draw.ellipse(bird_surf, (200, 200, 200), (6, wing_y, 14, 8), 1)

        # Rotate and blit
        rotated_bird = pygame.transform.rotate(bird_surf, angle)
        rot_rect = rotated_bird.get_rect(center=(int(bx + bw / 2.0), int(by + bh / 2.0)))
        self.screen.blit(rotated_bird, rot_rect.topleft)

        # 6. Score Overlay
        score_shadow = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
        score_surf = self.font.render(f"Score: {self.score}", True, COLOR_TEXT)
        self.screen.blit(score_shadow, (12, 12))
        self.screen.blit(score_surf, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        """Clean up pygame resources."""
        if self.render_mode:
            pygame.quit()

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

        if by + bh >= SCREEN_HEIGHT or by <= 0:
            return True

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
