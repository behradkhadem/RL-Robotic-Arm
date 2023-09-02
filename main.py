from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.urdf_common import UrdfEnv
import os
import gymnasium as gym
from reward import DistanceReward
from object import movable_obstacle


# Stable baselines 3
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from gymnasium.wrappers import FlattenObservation


robots = [
    GenericUrdfReacher(urdf="panda_with_gripper.urdf", mode="tor"),
]

render: bool = False
sensor = FullSensor(['position'], ['position', 'size'], variance=0.0)


from urdfenvs.wrappers.sb3_float32_action_wrapper import SB3Float32ActionWrapper
from normalize_action import MapActionWrapper
from gymnasium.wrappers import TimeLimit

roboticsEnv = UrdfEnv(render=render, robots=robots)
roboticsEnv.add_sensor(sensor, [0])
roboticsEnv.add_obstacle(movable_obstacle)
roboticsEnv.set_reward_calculator(DistanceReward())
roboticsEnv.set_spaces()
ob, *_ = roboticsEnv.reset()

roboticsEnv = FlattenObservation(roboticsEnv)
roboticsEnv = SB3Float32ActionWrapper(roboticsEnv)
roboticsEnv = MapActionWrapper(roboticsEnv)
roboticsEnv = TimeLimit(env=roboticsEnv, max_episode_steps= 250)

# roboticsEnv1 = UrdfEnv(render=render, robots=robots)
# roboticsEnv1.add_sensor(sensor, [0])
# roboticsEnv1.add_obstacle(movable_obstacle)
# roboticsEnv1.set_reward_calculator(DistanceReward())
# roboticsEnv1.set_spaces()
# ob, *_ = roboticsEnv1.reset()

# roboticsEnv1 = FlattenObservation(roboticsEnv1)
# roboticsEnv1 = SB3Float32ActionWrapper(roboticsEnv1)
# roboticsEnv1 = MapActionWrapper(roboticsEnv1)
# roboticsEnv1 = TimeLimit(env=roboticsEnv1, max_episode_steps= 250)



# env = gym.vector.SyncVectorEnv([
#     lambda: roboticsEnv, 
#     lambda: roboticsEnv1,
# ])

MODEL_NAME = 'TD3-001'
MODEL_CLASS = TD3

models_dir = 'models/' + MODEL_NAME
logdir = 'tb_logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


from stable_baselines3.common.callbacks import CheckpointCallback

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="./checkpoint/",
  name_prefix="robotic_arm_model",
  save_replay_buffer=False,
)


model = MODEL_CLASS("MlpPolicy", 
                    roboticsEnv, 
                    verbose=0, 
                    tensorboard_log= logdir, 
                    learning_rate= 0.001,
                    batch_size= 256, # It's better to be high. Default is 256. 
                    buffer_size= 2_000_000,
                    # gamma= 0.99, # Discount Factor 
                    # action_noise=action_noise, 
                    device='cuda',
                )

TIMESTEPS = 1_000_000

model.learn(
    total_timesteps=TIMESTEPS, 
    log_interval=10, 
    tb_log_name=MODEL_NAME,
    progress_bar=True, 
    callback= checkpoint_callback
)
