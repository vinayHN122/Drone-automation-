#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time
import torch

# Assuming env and ddqn are your local modules (make sure they are compatible with ROS2)
import env
import ddqn


class DroneAgentNode(Node):
    def __init__(self):
        super().__init__('drone_agent_node')

        # Initialize Gazebo UAV environment and agent
        self.GazeboUAV = env.GazeboUAV()
        self.agent = ddqn.DQN(self.GazeboUAV, batch_size=64, memory_size=10000,
                              target_update=4, gamma=0.99, learning_rate=1e-4,
                              eps=0, eps_min=0, eps_period=5000)

        param_path = '/home/yuhang/catkin_ws/src/uav_ros/scripts/Record/Duel_DQN_Reward_home2_sup.pth'
        self.agent.load_model(param_path, map_location=torch.device('cpu'))

        self.timer_period = 0.3  # seconds
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        self.state1 = None
        self.state2 = None
        self.step_count = 0

        self.reset_environment()

    def reset_environment(self):
        self.state1, self.state2 = self.GazeboUAV.reset()
        self.get_logger().info('Environment reset.')
        time.sleep(0.5)
        self.step_count = 0

    def control_loop(self):
        if self.state1 is None or self.state2 is None:
            self.reset_environment()

        action = self.agent.get_action(self.state1, self.state2)
        self.GazeboUAV.execute(action)
        time.sleep(0.1)  # small sleep to simulate real-time

        next_state1, next_state2, terminal, reward = self.GazeboUAV.step()
        self.step_count += 1

        if terminal or self.step_count >= 100:
            self.get_logger().info(f'Episode ended after {self.step_count} steps.')
            self.reset_environment()
        else:
            self.state1 = next_state1
            self.state2 = next_state2


def main(args=None):
    rclpy.init(args=args)
    node = DroneAgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
