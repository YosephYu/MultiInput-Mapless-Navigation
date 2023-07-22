from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from Env.vrepInterface_multi import CarNavi
import config
import os

name = f'{config.model}_{config.map}_{config.time}'
log_path = os.path.join(config.tb_log, name)
pth_path = os.path.join(config.ckpt, name)
env = CarNavi(config.DualVAE, config.device)
if not os.path.exists(log_path):
    os.makedirs(log_path)
new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
checkpoint_callback = CheckpointCallback(save_freq=config.ckpt_save_freq, save_path=pth_path,
                                        name_prefix=name)
model = SAC("MlpPolicy", env, batch_size=config.batchsize, learning_rate=config.lr, verbose=1)
model.set_logger(new_logger)
try:
    model.learn(total_timesteps=config.total_steps, log_interval=1, progress_bar=True, reset_num_timesteps=True, callback=checkpoint_callback)
except KeyboardInterrupt:
    model.save(pth_path)
else:
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.4f}, std_reward:{std_reward:.4f}")