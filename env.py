#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

import numpy as np
import cv2
import math
import random
import copy
from cv_bridge import CvBridge

from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point32
from sensor_msgs.msg import Image
from std_srvs.srv import Empty

import tf_transformations  # tf in ROS2 is via this package

import config


class GazeboUAV(Node):
    def __init__(self):
        super().__init__('gazebo_uav')

        # Parameters
        self.depth_image_size = [160, 120]
        self.bridge = CvBridge()
        self.vel_cmd = [0.0]

        self.default_states = None
        self.depth_image = None

        self.goal_space = config.goal_space
        self.start_space = config.start_space
        self.obstacle_pos = config.obstacle_position
        self.des = Point32()

        self.p = [21.0, 0.0]
        self.success = False
        self.dist_init = 0
        self.dist = 0
        self.reward = 0
        self.obstacle_state = []
        self._actions = []
        self.cylinder_pos = [[] for _ in range(10)]
        self.uav_trajectory = [[], []]
        self.stacked_imgs = None

        # Publishers
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.set_state_pub = self.create_publisher(ModelState, '/gazebo/set_model_state', 10)
        self.resized_depth_img_pub = self.create_publisher(Image, '/camera/depth/image_resized', 10)

        # Subscribers
        self.object_state_sub = self.create_subscription(ModelStates, 'gazebo/model_states', self.model_state_callback, 10)
        self.image_sub = self.create_subscription(Image, '/front_cam/camera/image', self.depth_image_callback, 10)

        # Services clients
        self.unpause_client = self.create_client(Empty, '/gazebo/unpause_physics')
        self.pause_client = self.create_client(Empty, '/gazebo/pause_physics')

        # Wait for services to be available
        while not self.unpause_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/unpause_physics service...')
        while not self.pause_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/pause_physics service...')

        # Small delay to settle subscriptions
        rclpy.spin_once(self, timeout_sec=1.0)

    def model_state_callback(self, msg: ModelStates):
        try:
            idx = msg.name.index("quadrotor")
        except ValueError:
            self.get_logger().warn("quadrotor model not found in gazebo/model_states")
            return

        pose = msg.pose[idx]
        twist = msg.twist[idx]

        quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        euler = tf_transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]

        self.self_state = [
            pose.position.x,
            pose.position.y,
            yaw,
            twist.linear.x,
            twist.linear.y,
            twist.angular.z,
        ]

        if self.default_states is None:
            self.default_states = copy.deepcopy(msg)

        for i in range(10):
            try:
                cyl_idx = msg.name.index("unit_cylinder" + str(i))
                self.cylinder_pos[i] = [msg.pose[cyl_idx].position.x, msg.pose[cyl_idx].position.y]
            except ValueError:
                self.get_logger().warn(f"unit_cylinder{i} model not found in gazebo/model_states")
                self.cylinder_pos[i] = [0.0, 0.0]

    def depth_image_callback(self, msg: Image):
        self.depth_image = msg

    def get_depth_image_observation(self):
        if self.depth_image is None:
            return None
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.depth_image, "bgr8")
            cv_img = np.array(cv_img, dtype=np.int8)
            cv_img[np.isnan(cv_img)] = 0
            return cv_img
        except Exception as err:
            self.get_logger().error(f"Ros_to_Cv2 Failure: {err}")
            return None

    def getdist(self):
        theta = self.self_state[2]
        a_x = self.des.x - self.self_state[0]
        a_y = self.des.y - self.self_state[1]
        c = math.sqrt(a_x ** 2 + a_y ** 2)
        alpha = math.atan2(a_y, a_x) - theta
        return c, alpha

    def goal2robot(self, d_x, d_y, theta):
        d = math.sqrt(d_x ** 2 + d_y ** 2)
        alpha = math.atan2(d_y, d_x) - theta
        return d, alpha

    def obstacle2robot(self, e_x, e_y):
        s_x = e_x - self.self_state[0]
        s_y = e_y - self.self_state[1]
        e = math.sqrt(s_x ** 2 + s_y ** 2)
        beta = math.atan2(s_x, s_y) - self.self_state[2]
        return e, beta

    def detect_collision(self):
        collision = False
        for i in range(len(self.cylinder_pos)):
            e, _ = self.obstacle2robot(self.cylinder_pos[i][0], self.cylinder_pos[i][1])
            if e < 1.2:
                collision = True
        return collision

    def get_states(self):
        if self.stacked_imgs is None:
            obs = self.get_depth_image_observation()
            if obs is None:
                return None, None
            self.stacked_imgs = np.dstack([obs] * 4)
        else:
            obs = self.get_depth_image_observation()
            if obs is None:
                return self.stacked_imgs, self.p
            self.stacked_imgs = np.dstack([self.stacked_imgs[:, :, -9:], obs])
        return self.stacked_imgs, self.p

    def get_actions(self):
        return self._actions

    def get_self_speed(self):
        return self.vel_cmd

    def set_uav_pose(self, x, y, theta):
        state = ModelState()
        state.model_name = 'quadrotor'
        state.reference_frame = 'world'
        # pose
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 1.5
        quaternion = tf_transformations.quaternion_from_euler(0, 0, theta)
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]
        # twist
        state.twist.linear.x = 0
        state.twist.linear.y = 0
        state.twist.linear.z = 0
        state.twist.angular.x = 0
        state.twist.angular.y = 0
        state.twist.angular.z = 0
        self.set_state_pub.publish(state)

    def set_object_pose(self):
        for i in range(10):
            state = ModelState()
            state.model_name = 'unit_cylinder' + str(i)
            state.reference_frame = 'world'
            state.pose.position.x = self.cylinder_pos[i][0]
            state.pose.position.y = self.cylinder_pos[i][1]
            state.pose.position.z = 1
            state.twist.linear.x = 0
            state.twist.linear.y = 0
            state.twist.linear.z = 0
            state.twist.angular.x = 0
            state.twist.angular.y = 0
            state.twist.angular.z = 0
            self.set_state_pub.publish(state)

    def set_object_pose_random(self):
        for i in range(10):
            state = ModelState()
            state.model_name = 'unit_cylinder' + str(i)
            state.reference_frame = 'world'
            state.pose.position.x = config.obstacle_position[i][0] + random.uniform(-1.0, 1.0)
            state.pose.position.y = config.obstacle_position[i][1] + random.uniform(-1.0, 1.0)
            state.pose.position.z = 1
            state.twist.linear.x = 0
            state.twist.linear.y = 0
            state.twist.linear.z = 0
            state.twist.angular.x = 0
            state.twist.angular.y = 0
            state.twist.angular.z = 0
            self.set_state_pub = self.create_publisher(ModelState, '/gazebo/set_model_state', 10)
            self.set_state_pub.publish(state)

    def set_goal(self, x, y):
        self.des.x = x
        self.des.y = y
        self.des.z = 1

    def reset(self):
        start_index = np.random.choice(len(self.start_space))
        goal_index = np.random.choice(len(self.goal_space))
        start = self.start_space[start_index]
        goal = self.goal_space[goal_index]

        theta = -math.pi / 2
        self.set_uav_pose(start[0], start[1], theta)
        self.set_goal(goal[0], goal[1])

        d0, alpha0 = self.goal2robot(goal[0] - start[0], goal[1] - start[1], theta)
        self.p = [d0, alpha0]
        self.reward = 0
        self.dist_init = d0
        self.vel_cmd = [0.0]
        self.success = False

        obs = self.get_depth_image_observation()
        if obs is None:
            # Handle the case if no image is received yet
            self.stacked_imgs = None
        else:
            self.stacked_imgs = np.dstack([obs] * 4)

        img, pos = self.get_states()
        return img, pos

    def execute(self, action_num):
        move_cmd = Twist()
        if action_num == 0:
            angular_z = 0.5
        elif action_num == 1:
            angular_z = 1.0
        elif action_num == 2:
            angular_z = -0.5
        elif action_num == 3:
            angular_z = -1.0
        elif action_num == 4:
            angular_z = 0
        else:
            raise Exception('Error discrete action')

        move_cmd.linear.x = 1.0
        move_cmd.angular.z = angular_z
        self.vel_pub.publish(move_cmd)

    def step(self):
        d1, alpha1 = self.getdist()
        self.p = [d1, alpha1]
        self.dist = d1
        terminal, reward = self.get_reward_and_terminate()
        self.reward = reward
        self.dist_init = self.dist
        img, pos = self.get_states()
        return img, pos, terminal, reward

    def get_reward_and_terminate(self):
        terminal = False
        reward = (10 * (self.dist_init - self.dist) - 0.2)

        if self.dist < 1:
            reward = 1000.0
            self.get_logger().info("Arrival!")
            terminal = True
            self.success = True

        if (self.self_state[0] >= 6.50 or self.self_state[0] <= -6.50 or
            self.self_state[1] >= 11.5 or self.self_state[1] <= -11.5):
            reward = -100.0
            self.get_logger().info("Out!")
            terminal = True

        if self.detect_collision():
            reward = -100.0
            self.get_logger().info("Collision!")
            terminal = True

        return terminal, reward
