#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" parameters setting """
import numpy as np
# path
DualVAE = "ckpt/DualVAE.ckpt"
model = "SAC"
map = "map1"
pth_map1 = f"ckpt/{model}_{map}_MultiInput.zip"
tb_log = './Exp/tb_log/'
ckpt = './Exp/ckpt/'
time = "1"
device = "cuda:0"

# the id of obstacles
if map == "map1":
    walls = [f"/wall[{i}]" for i in range(9)]       # map1
elif map == "map2":
    walls = [f"/wall[{i}]" for i in range(13)]      # map2

# the initial position
if map == "map1":
    pose = np.array([-1.8, -1.8, 15/180*np.pi])     # map1
elif map == "map2":
    pose = np.array([2, 1.23, 165/180*np.pi])       # map2

# setup
batchsize = 256
lr = 3e-4
ckpt_save_freq = int(1e4)
total_steps = int(1e6)
wait_response = False  # True: Synchronous response(too much delay)
n_ultra = 16
max_speed = 1
max_rotate = np.pi/2 * 0.5
time_step = 0.01

# coefficient
c_d = 200
c_p = 0.05
c_o = 0.02
c_l = 0.02

# threshold
eta_t = 0.3
eta_theta = 2.8
eta_l = 0.2

# reward
r_arrival = 150
r_collision = -50