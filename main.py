import time
from gridworld import CliffWalkingWapper, FrozenLakeWapper, GridWorld
from agent import QLearningAgent, SarsaAgent


def run_episode(env, agent, render=False, Map=None):
    total_steps = 0
    total_reward = 0

    obs = env.reset()

    Map = ''.join(str(i) for i in Map)

    while True:
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        if Map[next_obs] != 'G':
            reward -= 1
        if Map[next_obs] == 'H':
            reward -= 100
        if Map[next_obs] == 'G':
            reward += 1000

        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs
        total_reward += reward
        total_steps += 1
        if render:
            env.render()
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent, Map = None):
    total_reward = 0
    obs = env.reset()
    Map = ''.join(str(i) for i in Map)
    while True:
        action = agent.predict(obs)
        next_obs, reward, done, _ = env.step(action)
        if Map[next_obs] != 'G':
            reward -= 1
        if Map[next_obs] == 'H':
            reward -= 100
        if Map[next_obs] == 'G':
            reward += 1000
        total_reward += reward
        obs = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            print('test reward = %.1f' % (total_reward))
            break


def main():
    # env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up
    # env = FrozenLakeWapper(env)

    grid_map = ['SFFF', 'FHFH', 'FFFH', 'HFFG']
    env = GridWorld(grid_map)

    # env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    # env = CliffWalkingWapper(env)

    agent = QLearningAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.2)
    # agent = SarsaAgent(
    #     obs_n=env.observation_space.n,
    #     act_n=env.action_space.n,
    #     learning_rate=0.1,
    #     gamma=0.9,
    #     e_greed=0.1)

    is_render = False
    for episode in range(500):
        ep_reward, ep_steps = run_episode(env, agent, is_render, Map=grid_map)
        print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps,
                                                          ep_reward))

        # 每隔20个episode渲染一下看看效果
        if episode % 20 == 0:
            is_render = True
        else:
            is_render = False
    # 训练结束，查看算法效果
    test_episode(env, agent, Map=grid_map)


if __name__ == "__main__":
    main()
