from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.urdf_common import UrdfEnv
from panda.object import movable_obstacle
from panda.reward import DistanceReward
import numpy as np

from forwardkinematics.urdfFks.pandaFk import PandaFk


robots = [
    GenericUrdfReacher(urdf="panda_with_gripper.urdf", mode="tor"),
]

env = UrdfEnv(render=True, robots=robots)

sensor = FullSensor(['position'], ['position', 'size'], variance=0.0)
env.add_sensor(sensor, [0])
env.add_obstacle(movable_obstacle)
fk_panda = PandaFk()
env.set_reward_calculator(DistanceReward(fk_panda=fk_panda))

env.set_spaces()
ob, *_ = env.reset()

from urdfenvs.wrappers.sb3_float32_action_wrapper import SB3Float32ActionWrapper
from panda.normalize_action import MapActionWrapper
env = SB3Float32ActionWrapper(env)
# env = MapActionWrapper(env)

# You can interact inside the physics simulator window with the robot and move the arm itself. 
env.reset()
while True:
    # action = env.action_space.sample()
    # action = np.zeros_like(env.action_space.sample())
    action = env.action_space.sample() 
    obs, reward, done, truncated, info = env.step(action)