"""
train.py — Training loop for the DQN agent in the Flappy Bird environment.

Runs headless (no rendering) for maximum speed.
Periodically logs stats, saves checkpoints, and plots a score curve.

Usage:
    python train.py
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no window needed
import matplotlib.pyplot as plt

from game.flappy_env import FlappyEnv
from agent.dqn_agent import DQNAgent

# ─── Training hyperparameters ────────────────────────────────────────
NUM_EPISODES = 10_000        # Total training episodes
MAX_STEPS_PER_EP = 100_000  # Increased to prevent early capping of score
LOG_INTERVAL = 25            # Print summary every N episodes
SAVE_INTERVAL = 200          # Save checkpoint every N episodes
CHECKPOINT_DIR = "checkpoints"
PLOT_PATH = "training_curve.png"
RESUME_FROM = None   # Set to a checkpoint path to resume, e.g. "checkpoints/dqn_final.pth"

# Agent hyperparameters (passed through to DQNAgent)
AGENT_CONFIG = dict(
    lr=5e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_steps=1_000_000, # Increased exploration phase severely
    batch_size=64,
    buffer_capacity=100_000,
    target_update_freq=500,
    hidden_size=128,
)


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    env = FlappyEnv(render_mode=False)   # Headless — no pygame display
    agent = DQNAgent(**AGENT_CONFIG)

    # ── Resume from checkpoint ───────────────────────────────────────
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        agent.load(RESUME_FROM)
        print(f"▶ Resuming training from {RESUME_FROM}")
    else:
        print("▶ Starting fresh training run")
    print(f"🚀 Training on device: {agent.device}")
    print(f"   Episodes: {NUM_EPISODES}  |  Epsilon decay over {AGENT_CONFIG['epsilon_decay_steps']} steps")
    print("=" * 65)

    # ── Tracking ─────────────────────────────────────────────────────
    all_scores: list[int] = []
    all_rewards: list[float] = []
    best_score = 0
    start_time = time.time()

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(MAX_STEPS_PER_EP):
            action = agent.select_action(state, training=True)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        score = env.score
        all_scores.append(score)
        all_rewards.append(total_reward)

        # Track best
        if score > best_score:
            best_score = score
            agent.save(os.path.join(CHECKPOINT_DIR, "dqn_best.pth"))

        # ── Periodic logging ─────────────────────────────────────────
        if episode % LOG_INTERVAL == 0:
            recent = all_scores[-LOG_INTERVAL:]
            avg = np.mean(recent)
            mx = np.max(recent)
            elapsed = time.time() - start_time
            print(
                f"Ep {episode:>5} | "
                f"ε {agent.epsilon:.4f} | "
                f"Avg score {avg:>6.1f} | "
                f"Max {mx:>3} | "
                f"Best ever {best_score:>3} | "
                f"Steps {steps:>4} | "
                f"Time {elapsed:>6.0f}s"
            )

        # ── Periodic checkpoint ──────────────────────────────────────
        if episode % SAVE_INTERVAL == 0:
            agent.save(os.path.join(CHECKPOINT_DIR, f"dqn_ep{episode}.pth"))

    # ── Final save and plot ──────────────────────────────────────────
    agent.save(os.path.join(CHECKPOINT_DIR, "dqn_final.pth"))
    env.close()

    _plot_training_curve(all_scores)
    print("=" * 65)
    print(f"✅ Training complete!  Best score: {best_score}")
    print(f"   Final checkpoint : {CHECKPOINT_DIR}/dqn_final.pth")
    print(f"   Best  checkpoint : {CHECKPOINT_DIR}/dqn_best.pth")
    print(f"   Score curve      : {PLOT_PATH}")


def _plot_training_curve(scores: list[int], window: int = 50):
    """Save a smoothed training-score plot to disk."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.3, color="steelblue", label="Episode score")
    if len(scores) >= window:
        avg = np.convolve(scores, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(scores)), avg, color="tomato",
                 linewidth=2, label=f"{window}-ep moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Training Score")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(scores, alpha=0.3, color="steelblue")
    if len(scores) >= window:
        plt.plot(range(window - 1, len(scores)), avg, color="tomato", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Training Score (zoomed)")
    if len(scores) > 200:
        plt.xlim(len(scores) - 500, len(scores))
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"📊 Training curve saved to {PLOT_PATH}")


if __name__ == "__main__":
    train()
