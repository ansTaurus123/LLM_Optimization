import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv

# Define a custom Gym environment for RL-based hallucination detection
class HallucinationEnv(gym.Env):
    def __init__(self, df):
        super(HallucinationEnv, self).__init__()
        self.df = df
        self.index = 0
        self.action_space = gym.spaces.Discrete(2)  # 0 = Factual, 1 = Hallucinated
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)  # Confidence, Entropy

    def step(self, action):
        """Reward: Correct classification → +1, Incorrect → -1"""
        hallucinated = int(self.df.iloc[self.index]["hallucinated_response"] != self.df.iloc[self.index]["correct_answer"])
        reward = 1 if action == hallucinated else -1
        self.index = (self.index + 1) % len(self.df)
        return self._get_obs(), reward, False, {}

    def _get_obs(self):
        """Extracts confidence & entropy as state representation"""
        return np.array([np.random.rand(), np.random.rand()])  # Placeholder (replace with real model metrics)

    def reset(self):
        self.index = 0
        return self._get_obs()

# Train the RL Model
env = DummyVecEnv([lambda: HallucinationEnv(df)])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("hallucination_detector_rl")
