# Autonomous Game Agent: Deep Reinforcement Learning in Flappy Bird

A modular, production-grade implementation of an autonomous AI agent trained to master Flappy Bird from scratch using **Deep Q-Networks (DQN)** in **PyTorch**. 

Instead of processing raw pixels, which introduces severe computational overhead, this project utilizes custom **feature engineering** to map a highly efficient, 4-dimensional continuous state space. The system features an entirely decoupled architecture, separating the client-side physics engine from the machine learning backend pipelines.

---

## 🚀 Key Project Architecture

The codebase is organized into highly isolated modules following standard machine learning and software engineering separation of concerns:

```text
├── game/
│   ├── bird.py          # Physics engine for the agent (gravity, flight forces, bounding boxes)
│   └── pipe.py          # Procedural obstacle engine (randomized height gaps, coordinate displacement)
├── agent/
│   └── dqn_agent.py     # DQN neural network, replay buffer matrices, and epsilon-greedy policies
├── flappy_env.py        # Gym-style wrapper environment (step, reset, reward mapping API)
├── train.py             # Optimization and backpropagation pipeline
└── evaluate.py          # Inference script for visual model evaluation and live human vs. AI testing
