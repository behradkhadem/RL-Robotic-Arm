from urdfenvs.urdf_common.reward import Reward

import numpy as np
from forwardkinematics.urdfFks.pandaFk import PandaFk

class DistanceReward(Reward):
    def calculate_reward(self, observation: dict) -> float:
        target_position = np.array([0.3, -0.4, 0.02])

        obstacle_position = observation["robot_0"]["FullSensor"]["obstacles"][1]["position"]
        joint_position = observation['robot_0']['joint_state']['position'][:7]
        fk_panda = PandaFk()
        ee_position = fk_panda.fk(joint_position, 7, positionOnly=True)

        distance_to_endeffector = np.linalg.norm(obstacle_position - ee_position)

        distance_to_target = np.linalg.norm(target_position - obstacle_position)

        reward = (1 / distance_to_target) * 10  + (1 / distance_to_endeffector * 1)

        print(f"🎁 Reward: {reward}")
        return reward.astype(int)
    