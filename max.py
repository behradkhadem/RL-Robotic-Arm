from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.urdf_common import UrdfEnv
from urdfenvs.urdf_common.reward import Reward
from mpscenes.obstacles.box_obstacle import BoxObstacle

import numpy as np
import os
    
class DistanceReward(Reward):
    def calculate_reward(self, observation: dict) -> float:

        endeffector_position = observation['robot_0']['joint_state']['position'][:3] # first three params of the position array.
        print(f"🦾 End-effector Position: {endeffector_position}")
        obstacle_position = observation["robot_0"]["FullSensor"]["obstacles"][1]["position"]
        print(f'⬛ Obstacle Position: {obstacle_position}')
        reward = 0.0
        return reward
    
movable_obstacle_dict = {
'type': 'box',
'geometry': {
    'position' : [0.5, 0.5, 0.0],
    'width': 0.04,
    'height': 0.04,
    'length': 0.1,
},
'movable': True,
'high': {
        'position' : [5.0, 5.0, 1.0],
    'width': 0.35,
    'height': 0.35,
    'length': 0.35,
},
'low': {
    'position' : [0.0, 0.0, 0.5],
    'width': 0.2,
    'height': 0.2,
    'length': 0.2,
}
}
movable_obstacle = BoxObstacle(name="movable_box", content_dict=movable_obstacle_dict)


robots = [
    GenericUrdfReacher(urdf="panda_with_gripper.urdf", mode="vel"),
]

env = UrdfEnv(render=True, robots=robots)

sensor = FullSensor(['position'], ['position', 'size'], variance=0.0)
env.add_sensor(sensor, [0])
env.add_obstacle(movable_obstacle)
env.set_reward_calculator(DistanceReward())

env.set_spaces()
ob, *_ = env.reset()

# You can interact inside the physics simulator window with the robot and move the arm itself. 
env.reset()
while True:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)