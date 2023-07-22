#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" vrep interface script that connects to vrep, reads and sets data to objects through vrep remote API  """

import time
import numpy as np
import Env.sim as vrep
import config
import math
import gym
from gym import spaces
from Env.VAE import DualVAE
import torch

# V-REP data transmission modes:
WAIT = vrep.simx_opmode_oneshot_wait
ONESHOT = vrep.simx_opmode_oneshot
STREAMING = vrep.simx_opmode_streaming
BUFFER = vrep.simx_opmode_buffer


if config.wait_response:
    MODE_INI = WAIT
    MODE = WAIT
else:
    MODE_INI = STREAMING
    MODE = BUFFER


class CarNavi(gym.Env):
    def __init__(self, path=None, device=None) -> None:
        """ Connect to the simulator"""
        super(CarNavi, self).__init__()
        self.action_space = spaces.Box(
            low=np.array((-1, -1)), high=np.array((1, 1)), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(35,), dtype=np.float32)
        self.robotID = -1
        self.ultraID = [-1] * config.n_ultra
        self.rgbID = -1
        self.depthID = -1
        self.wallsID = [-1] * len(config.walls)
        self.targetID = -1
        self.left_motorID = -1
        self.right_motorID = -1
        self.prev_dist = 0
        self.encoder = DualVAE(32)
        if path:
            self.encoder.load_state_dict(torch.load(path))
        if device:
            self.encoder = self.encoder.to(device=device)
        ip = '127.0.0.1'
        port = 19997
        vrep.simxFinish(-1)
        self.clientID = vrep.simxStart(ip, port, True, True, 3000, 5)
        # Connect to V-REP
        if self.clientID == -1:
            import sys
            sys.exit('\nV-REP remote API server connection failed (' + ip + ':' +
                     str(port) + '). Is V-REP running?')
        print('Connected to Remote API Server')  # show in the terminal
        self.start()

    def reset(self):
        """ reset the position of the robot"""
        pose = config.pose
        self.set_robot_pose2d(pose, relative=False)
        rgb, depth = self.get_rgbd()
        rgbd = self.encoder(rgb, depth)
        self.prev_dist = np.linalg.norm(pose[:2])
        phi = np.arctan2(pose[1], pose[0])
        state = np.concatenate([np.array([self.prev_dist/5, phi/np.pi, pose[2]/np.pi]), rgbd])
        return state

    def step(self, action):
        self.move_wheels(action[0], action[1])
        rgb, depth = self.get_rgbd()
        pose = self.get_robot_pose2d(relative=True)
        collision = self.check_collision()
        dist = np.linalg.norm(pose[:2])
        phi = np.arctan2(pose[1], pose[0])
        reward = self.get_reward(collision=collision, dist=dist, phi=phi, alpha=pose[2])
        # print(f'reward:{reward:3.4f}')
        self.prev_dist = dist
        done = False
        if collision or (dist < config.eta_t):
            done = True
        rgbd = self.encoder(rgb, depth)
        state = np.concatenate([np.array([dist/5, phi/np.pi, pose[2]/np.pi]), rgbd])
        return state, reward, done, {}

    def render(self):
        pass

    def close(self):
        """ Disconnect from the simulator"""
        # Make sure that the last command sent has arrived
        vrep.simxGetPingTime(self.clientID)
        # Now close the connection to V-REP:
        vrep.simxFinish(self.clientID)

    def start(self):
        """ Start the simulation (force stop and setup)"""
        vrep.simxStopSimulation(self.clientID, ONESHOT)
        self.setup_devices()
        time.sleep(0.5)
        vrep.simxStartSimulation(self.clientID, ONESHOT)

    def setup_devices(self):
        """ Assign the devices from the simulator to specific IDs """
        # robot
        _, self.robotID = vrep.simxGetObjectHandle(
            self.clientID, '/robot', WAIT)
        # motors
        _, self.left_motorID = vrep.simxGetObjectHandle(
            self.clientID, '/robot/leftMotor', WAIT)
        _, self.right_motorID = vrep.simxGetObjectHandle(
            self.clientID, '/robot/rightMotor', WAIT)
        # ultrasonic sensors
        for idx in range(config.n_ultra):
            _, self.ultraID[idx] = vrep.simxGetObjectHandle(
                self.clientID, f"/robot/ultra[{idx}]", WAIT)
        #rgbd
        _, self.rgbID = vrep.simxGetObjectHandle(self.clientID, '/robot/kinect/rgb', WAIT)
        _, self.depthID = vrep.simxGetObjectHandle(self.clientID, '/robot/kinect/depth', WAIT)
        vrep.simxGetVisionSensorImage(self.clientID, self.rgbID, 0, STREAMING)
        vrep.simxGetVisionSensorDepthBuffer(self.clientID, self.depthID, STREAMING)
        # obstacles
        for idx, wall in enumerate(config.walls):
            _, self.wallsID[idx] = vrep.simxGetObjectHandle(
                self.clientID, wall, WAIT)
        # target
        _, self.targetID = vrep.simxGetObjectHandle(
            self.clientID, '/Target', WAIT)

        # start up devices

        # wheels
        vrep.simxSetJointTargetVelocity(
            self.clientID, self.left_motorID, 0, STREAMING)
        vrep.simxSetJointTargetVelocity(
            self.clientID, self.right_motorID, 0, STREAMING)
        # pose
        vrep.simxGetObjectPosition(self.clientID, self.robotID, -1, MODE_INI)
        vrep.simxGetObjectOrientation(self.clientID, self.robotID, -1, MODE_INI)
        # relative pose
        vrep.simxGetObjectPosition(self.clientID, self.robotID, self.targetID, MODE_INI)
        vrep.simxGetObjectOrientation(self.clientID, self.robotID, self.targetID, MODE_INI)
        # reading-related function initialization according to the recommended operationMode
        for i in self.ultraID:
            vrep.simxReadProximitySensor(self.clientID, i, STREAMING)

        for i in self.wallsID:
            vrep.simxCheckCollision(self.clientID, self.robotID, i, STREAMING)

    def get_robot_pose2d(self, relative=True):
        """ return the pose of the robot:  [ x(m), y(m), Theta(rad) ] """
        if relative:
            _, pos = vrep.simxGetObjectPosition(
                self.clientID, self.robotID, self.targetID, MODE)
            _, ori = vrep.simxGetObjectOrientation(
                self.clientID, self.robotID, self.targetID, MODE)
        else:
            _, pos = vrep.simxGetObjectPosition(
                self.clientID, self.robotID, -1, MODE)
            _, ori = vrep.simxGetObjectOrientation(
                self.clientID, self.robotID, -1, MODE)
        pos = np.array([pos[0], pos[1], ori[2]])
        return pos

    def set_robot_pose2d(self, pose, relative=True):
        """ set the pose of the robot:  [ x(m), y(m), Theta(rad) ] """
        if relative:
            vrep.simxSetObjectPosition(self.clientID, self.robotID, self.targetID, [
                                       pose[0], pose[1], 0.1187], WAIT)
            vrep.simxSetObjectOrientation(self.clientID, self.robotID, self.targetID, [
                                          0, 0, pose[2]], WAIT)
        else:
            vrep.simxSetObjectPosition(
                self.clientID, self.robotID, -1, [pose[0], pose[1], 0.1388], WAIT)
            vrep.simxSetObjectOrientation(
                self.clientID, self.robotID, -1, [0, 0, pose[2]], WAIT)

    def get_ultra_distance(self):
        """ return distances measured by ultrasonic sensors(m) """
        distances = np.ones(config.n_ultra, dtype=np.float32)
        # angles = np.ones_like(distances) * (-1)
        for i, item in enumerate(self.ultraID):
            _, state, detectedPoint, _, _ = vrep.simxReadProximitySensor(
                self.clientID, item, BUFFER)
            if state == True:
                distances[i] = math.sqrt(detectedPoint[0]**2 + detectedPoint[1]**2 + detectedPoint[2]**2)/3 # divide 3 for normalization
        return distances
    
    def get_rgbd(self):
        """ get the rgb image and depth image """
        _, resolution, rgb_raw = vrep.simxGetVisionSensorImage(self.clientID, self.rgbID, 0, BUFFER)
        _, resolution, depth_raw = vrep.simxGetVisionSensorDepthBuffer(self.clientID, self.depthID, BUFFER)
        rgb = np.array(rgb_raw, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
        depth = np.array(depth_raw, dtype=np.float32).reshape(resolution[1], resolution[0])
        return rgb, depth


    def move_wheels(self, alpha, beta):
        """ move the wheels. Input: Angular velocities in rad/s """
        alpha = 0.5*(alpha+1)  # map the action[0] from [-1,1] to [0,1]
        # map the linear and angular velocities to wheels rotation speed
        v_right = (2*alpha*config.max_speed + beta*config.max_rotate)/2
        v_left = (2*alpha*config.max_speed - beta*config.max_rotate)/2
        vrep.simxSetJointTargetVelocity(
            self.clientID, self.left_motorID, v_left, STREAMING)
        vrep.simxSetJointTargetVelocity(
            self.clientID, self.right_motorID, v_right, STREAMING)
        time.sleep(config.time_step)
        return

    def check_collision(self):
        """ judge if collision happens"""
        for wall in self.wallsID:
            _, collision = vrep.simxCheckCollision(
                self.clientID, self.robotID, wall, BUFFER)
            if collision:
                return True
        return False

    def stop_motion(self):
        """ stop the base wheels """
        vrep.simxSetJointTargetVelocity(
            self.clientID, self.left_motorID, 0, STREAMING)
        vrep.simxSetJointTargetVelocity(
            self.clientID, self.right_motorID, 0, STREAMING)

    def get_reward(self, collision, dist, phi=0, alpha=0, min_dist=5):
        r_ori = 0
        angle_delta = np.abs(phi-alpha)
        if angle_delta > config.eta_theta and angle_delta <= np.pi:
            r_ori = config.c_o * angle_delta

        r_laser = 0
        if min_dist > 0 and min_dist < config.eta_l:
            r_laser = - config.c_l / min_dist

        dist_delta = self.prev_dist - dist
        reward = config.c_d * dist_delta - config.c_p + r_ori + r_laser
        if collision:
            reward = config.r_collision
        if dist < config.eta_t:
            reward = config.r_arrival
        return reward
