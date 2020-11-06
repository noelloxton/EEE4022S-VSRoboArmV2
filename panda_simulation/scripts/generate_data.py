#! /usr/bin/env python
# -*- coding: utf-8 -*-

#uni: University of Cape Town
#author: Noel Loxton
#student ID: LXTNOE001
#This code forms part of the EEE4022S honours thesis project, my project is Visual Servoing of a Simulated Robotic Arm
#This file is used to generate datasets from the ROS simulation


#import os for running bash commands
import os

#Define global constants for the target start position on target spawn 
START_target_X = 0.76
START_target_Y = 0.47
START_target_Z = 0.7974
#Define global constants for the max target and arm parameters 
MAX_arm_Z = 1.15
MIN_arm_Z = 0.2
MAX_target_Z = 1.2724
MAX_arm_Y = START_target_Y
MIN_arm_Y = -0.5
MIN_target_Y = MIN_arm_Y
#Define target increments
#disp_Z = MAX_target_Z - START_target_Z
#no_incr = 20 #define number of increments here - this is also how many datapoints/images will be made for training/verification
#incr = disp_Z/(no_incr-1) #this splits the total z-displacment into increments for the target to move to
disp_Y = START_target_Y - MIN_target_Y
no_incr = 100 #define number of increments here - this is also how many datapoints/images will be made for training/verification
incr = disp_Y/(no_incr-1) #this splits the total z-displacment into increments for the target to move to

#Some of the code in the next section is taken from this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
# Sys
import sys

# Instantiate CvBridge
bridge = CvBridge()

def image_callback(msg):
    print("Received an image! Waiting...")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg
        cv2.imwrite('camera_image.jpeg', cv2_img)

def save_img():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/panda_arm/camera1/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.sleep(2)
    #rospy.spin()
    #sys.exit()

#function to create the text file to store labels for data
def create_file():
    f = open("label.txt","w+")
    f.close()

#function to add labels to the 'label' text file
def add_label(z):
    f = open("label.txt", "a+")
    f.write(str(z)+"\n")
    f.close()
    
#function to create the images from the ROS simulation
def create_images():

    #new_Z = START_target_Z #this is used for incrementing target's Z-coord
    new_Y = START_target_Y #this is used for incrementing target's Y-coord
    
    #loop number of images you want to save (coincides to # increments)
    for x in range(no_incr):

    	#spawn target into gazebo
	spawnCommand = "rosrun gazebo_ros spawn_model -file ~/catkin_ws/src/arm_description/models/qr_code_target/model.sdf -sdf -model target -x {} -y {} -z {}".format(START_target_X, new_Y, START_target_Z)
	os.system(spawnCommand)
	rospy.sleep(2) #sleep for 1/2s to ensure only 1 image saved
	#save image and rename it
        save_img()
	rospy.sleep(2) #sleep for 1/2s to ensure only 1 image saved
	img_name = "test"+str(x) #rename img
        renameCommand = "mv ~/panda_arm_sim/src/panda_simulation/data_for_nn/camera_image.jpeg ~/panda_arm_sim/src/panda_simulation/data_for_nn/{}.jpeg".format(img_name) 
	os.system(renameCommand)
	
	#make a file with the title of the z-coord
	#makeCommand = "mkdir {}".format(new_Z-0.1224)#for z
        #make a file with the title of the y-coord
        makeCommand = "mkdir -- {}".format(new_Y)       #for y
	os.system(makeCommand)
	#move the img to the file
	#moveCommand = "mv {}.jpeg ~/panda_arm_sim/src/panda_simulation/data_for_nn/{}".format(img_name,new_Z-0.1224) #for z
        moveCommand = "mv {}.jpeg ~/panda_arm_sim/src/panda_simulation/data_for_nn/{}".format(img_name,new_Y)        #for y
	os.system(moveCommand)
	
	#add the corresponding z-coord to the label file
	#add_label((new_Z - 0.1224))#for Z
        #add the corresponding y-coord to the label file
	add_label((new_Y))         #for Y

	#increment z-coord
	#new_Z += incr
        #increment y-coord
	new_Y -= incr

	#delete model
	deleteCommand = "rosservice call gazebo/delete_model '{model_name: target}'"
	os.system(deleteCommand)
	
#main function
def main():
    create_file()
    create_images()
    print("test sample")
    print("is it over yet")
    print("SURELY")
    #deleteCommand = "rm -r ~/arm_testing_ws/src/panda_simulation/data_for_nn/camera_image.jpeg"
    #os.system(deleteCommand)
    
if __name__ == '__main__':
    main()

