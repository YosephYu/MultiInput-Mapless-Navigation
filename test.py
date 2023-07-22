from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from Env.vrepInterface_multi import CarNavi
import config as config

env = CarNavi(config.DualVAE, config.device)
model = SAC.load(config.pth_map1, env=env)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=2)
print(f"mean_reward:{mean_reward:.4f}, std_reward:{std_reward:.4f}")

print("============Test one epoch============")
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
env.stop_motion()
print("===============Success===============")
env.close()
