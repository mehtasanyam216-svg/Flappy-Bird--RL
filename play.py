import pygame
from game.flappy_env import FlappyEnv
from agent.dqn_agent import DQNAgent

env = FlappyEnv(render_mode=True)
agent = DQNAgent()

agent.load("checkpoints/dqn_best.pth")
agent.epsilon = 0.0

state = env.reset()
done = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            exit()

    if not done:
        action = agent.select_action(state, training=False)
        state, _, done = env.step(action)
    else:
        pygame.time.wait(1000)
        state = env.reset()
        done = False

    env.render()
