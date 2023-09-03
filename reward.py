from urdfenvs.urdf_common.reward import Reward
import numpy as np
from forwardkinematics.urdfFks.pandaFk import PandaFk

class DistanceReward(Reward):
    def __init__(self, fk_panda) -> None:
        self.fk = fk_panda
        super().__init__()


    def calculate_reward(self, observation: dict) -> float:
        target_position = np.array([0.3, -0.4, 0.02])

        obstacle_position = observation["robot_0"]["FullSensor"]["obstacles"][1]["position"]
        joint_position = observation['robot_0']['joint_state']['position'][:7]
        # fk_panda = PandaFk()
        fk_panda = self.fk
        ee_position = fk_panda.numpy(joint_position, 7, positionOnly=True)
        # print(f'🛠️ ee_position: {ee_position}')

        distance_to_endeffector = np.linalg.norm(obstacle_position - ee_position)

        distance_to_target = np.linalg.norm(target_position - obstacle_position)

        # reward = (1 / distance_to_target) * 10  + (1 / distance_to_endeffector * 1)

        reward = -10 * distance_to_target - distance_to_endeffector

        print(f"🎁 Reward: {reward}")
        return reward.astype(int)
    