"""
evaluate.py — Load a trained DQN checkpoint and watch the agent play Flappy Bird.

Usage:
    python evaluate.py                            
    python evaluate.py checkpoints/dqn_ep1000.pth 
"""
import sys
import pygame
from game.flappy_env import FlappyEnv
from agent.dqn_agent import DQNAgent

DEFAULT_CHECKPOINT = "checkpoints/dqn_best.pth"
NUM_EPISODES = 5

def evaluate(checkpoint_path: str):
    env = FlappyEnv(render_mode=True)
    agent = DQNAgent()
    agent.load(checkpoint_path)

    print(f"\n▶ Watching agent play ({NUM_EPISODES} episodes)...\n")

    for ep in range(1, NUM_EPISODES + 1):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    sys.exit()

            action = agent.select_action(state, training=False)
            state, reward, done = env.step(action)
            total_reward += reward
            env.render()

        print(f"Episode {ep}: Score = {env.score}  Total reward = {total_reward:+.1f}")
        pygame.time.wait(800)

    env.close()
    print("\n✅ Evaluation complete.")

import random

def random_agent_test(env, episodes=5):
    print("\n▶ Running random agent for comparison...\n")

    scores = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False

        while not done:
            action = random.choice([0, 1])
            state, _, done = env.step(action)

        scores.append(env.score)
        print(f"[Random] Episode {ep}: Score = {env.score}")

    avg = sum(scores) / len(scores)
    print(f"\n🎯 Random Agent Average Score: {avg:.2f}\n")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
    env = FlappyEnv(render_mode=False)

# Random baseline
    random_agent_test(env)

    env.close()

# Trained agent
    evaluate(path)
