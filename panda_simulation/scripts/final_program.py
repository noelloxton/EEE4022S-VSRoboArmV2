#! /usr/bin/env python
# -*- coding: utf-8 -*-

#uni: University of Cape Town
#author: Noel Loxton
#student ID: LXTNOE001
#This code forms part of the EEE4022S honours thesis project, my project is Visual Servoing of a Simulated Robotic Arm
#This file acts as the full end-to-end vs system


#import neccessary libraries and modules
"""
import tensorflow as tf
import keras
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skimage.transform as sktransform

from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling
from keras.models import load_model
"""

import os
#launch the arm-camera system in the background
launchCommand = "cd ~/panda_arm_sim/src/panda_simulation/scripts && ./launch_sim.sh && sleep 20"
os.system(launchCommand)


#set the arm to the correct position
moveCommand = "cd ~/panda_arm_sim/src/panda_simulation/scripts && ./move_arm.sh && sleep 7"
os.system(moveCommand)

#get user input
user_in_z = 0
while ((user_in_z < 0.3224) or (user_in_z >1.2724)): 
  try:
    user_in_z = input("Enter a z-coordinate spawn position for the target (Please Enter in the range: 0.3224 <= z <= 1.2724 ): ")
    user_in_z = float(user_in_z)
  except ValueError:
    print("Error: that's not a valid input:(. Please try again")

#spawn target at user location
spawnCommand = "rosrun gazebo_ros spawn_model -file ~/catkin_ws/src/arm_description/models/qr_code_target/model.sdf -sdf -model target -x 0.2 -y 0 -z {}".format(user_in_z)
os.system(spawnCommand)

#open image window
launchImage = "cd ~/panda_arm_sim/src/panda_simulation/scripts && ./launch_image_window.sh"
os.system(launchImage)

virtualCommand = "workon dl4cv"
os.system(virtualCommand)

#save image to file system
#os.system("cd ~/panda_arm_sim/src/panda_simulation/final_ds && rosrun panda_simulation ros_image_saver")





