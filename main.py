#!/usr/bin/env python3
import os
import time
import numpy as np
import torch
import rclpy
from rclpy.node import Node

import env
import ddqn


class DroneTrainerNode(Node):
    def __init__(self):
        super().__init__('drone_trainer_node')

        self.GazeboUAV = env.GazeboUAV()
        self.agent = ddqn.DQN(
            self.GazeboUAV,
            batch_size=64,
            memory_size=10000,
            target_update=4,
            gamma=0.99,
            learning_rate=1e-4,
            eps=0.95,
            eps_min=0.1,
            eps_period=5000,
            network='DQN'
        )

        self.model_path = '/home/zyh/catkin_ws/src/Uav/scripts/Record/'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.total_episode = 15000
        self.max_step_per_episode = 70
        self.ep_reward_list = []

        # Start training immediately
        self.train()

    def train(self):
        for i_episode in range(self.total_episode + 1):
            if (i_episode % 10 == 0 and i_episode != 0):
                self.GazeboUAV.SetObjectPose_random()
            else:
                self.GazeboUAV.SetObjectPose()

            state1, state2 = self.GazeboUAV.reset()
            time.sleep(0.5)

            ep_reward = 0
            for t in range(self.max_step_per_episode):
                action = self.agent.get_action(state1, state2)
                self.GazeboUAV.execute(action)
                ts = time.time()

                if len(self.agent.replay_buffer.memory) > 64:
                    self.agent.learn()

                while time.time() - ts <= 0.5:
                    pass

                next_state1, next_state2, terminal, reward = self.GazeboUAV.step()
                ep_reward += reward

                self.agent.replay_buffer.add(state1, state2, action, reward,
                                            next_state1, next_state2, terminal)

                if terminal:
                    break

                state1 = next_state1
                state2 = next_state2

            self.ep_reward_list.append(ep_reward)
            self.get_logger().info(
                f"Episode:{i_episode} step:{t} ep_reward:{ep_reward} epsilon:{round(self.agent.eps, 4)}"
            )

            if (i_episode + 1) % 100 == 0:
                np.savetxt(self.model_path + 'DQN_home_sup.txt', self.ep_reward_list)
                self.agent.save_model(self.model_path + 'DQN_home_sup.pth')


def main(args=None):
    rclpy.init(args=args)
    node = DroneTrainerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
