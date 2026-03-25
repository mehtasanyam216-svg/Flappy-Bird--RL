"""
main.py — Runner script for the Flappy Bird RL environment.

Modes:
    HUMAN_MODE = True   →  Spacebar controls the bird (manual play).
    HUMAN_MODE = False  →  A random-action placeholder agent drives the bird.

In both modes, state / reward / done are printed each frame for debugging.
"""
import sys
import random
import pygame
from game.flappy_env import FlappyEnv

HUMAN_MODE = True

def run_human(env: FlappyEnv):
    """Manual play loop controlled by the spacebar."""
    state = env.reset()
    done = False

    while True:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                action = 1

        if not done:
            state, reward, done = env.step(action)
            print(f"State: {[round(s, 2) for s in state]}  "
                  f"Reward: {reward:+.1f}  Done: {done}  "
                  f"Score: {env.score}")

        if done:
            env.render()
            pygame.time.wait(500)
            state = env.reset()
            done = False

        env.render()

def run_agent(env: FlappyEnv):
    """Placeholder agent loop — takes random actions for demonstration."""
    episodes = 5
    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    sys.exit()

            action = random.choice([0, 0, 0, 1])
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

            env.render()

            if steps % 30 == 0:
                print(f"[Ep {ep}] State: {[round(s, 2) for s in state]}  "
                      f"Reward: {reward:+.1f}  Done: {done}")

        print(f"═══ Episode {ep} finished — Score: {env.score}  "
              f"Total reward: {total_reward:+.1f}  Steps: {steps} ═══\n")

    env.close()

if __name__ == "__main__":
    env = FlappyEnv(render_mode=True)

    if HUMAN_MODE:
        print("▶ HUMAN MODE — Press SPACE to flap!")
        run_human(env)
    else:
        print("▶ AGENT MODE — Random actions (placeholder)")
        run_agent(env)
