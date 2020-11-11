#! /usr/bin/env python
# -*- coding: utf-8 -*-

#To use the Python MoveIt interfaces, we will import the moveit_commander namespace. This namespace provides us with a MoveGroupCommander class, a PlanningSceneInterface class, and a RobotCommander class. We also import rospy and some messages that we will use:
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

#First initialize moveit_commander and a rospy node:
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

#Instantiate a RobotCommander object. Provides information such as the robot’s kinematic model and the robot’s current joint states
robot = moveit_commander.RobotCommander()

#Instantiate a PlanningSceneInterface object. This provides a remote interface for getting, setting, and updating the robot’s internal understanding of the surrounding world:
scene = moveit_commander.PlanningSceneInterface()

#Instantiate a MoveGroupCommander object. This object is an interface to a planning group (group of joints).This interface can be used to plan and execute motions:
group_name = "panda_arm"
move_group = moveit_commander.MoveGroupCommander(group_name)

#Create a DisplayTrajectory ROS publisher which is used to display trajectories in Rviz:
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)

# We can get the name of the reference frame for this robot:
planning_frame = move_group.get_planning_frame()
print("============ Planning frame: %s" % planning_frame)

# We can also print the name of the end-effector link for this group:
eef_link = move_group.get_end_effector_link()
print("============ End effector link: %s" % eef_link)

# We can get a list of all the groups in the robot:
group_names = robot.get_group_names()
print("============ Available Planning Groups:", robot.get_group_names())

# Sometimes for debugging it is useful to print the entire state of the
# robot:
print("============ Printing robot state")
print(robot.get_current_state())
print("")

print("=========== Printing current pose")
print(move_group.get_current_pose().pose)
print("")

"""
# We can get the joint values from the group and adjust some of the values:
joint_goal = move_group.get_current_joint_values()
joint_goal[0] = -0.6
joint_goal[1] = 0.6
joint_goal[2] = 0.75
joint_goal[3] = -1.5
joint_goal[4] = -0.4
joint_goal[5] = 2
joint_goal[6] = -2

# The go command can be called with joint values, poses, or without any
# parameters if you have already set the pose or joint target for the group
move_group.go(joint_goal, wait=True)

# Calling ``stop()`` ensures that there is no residual movement
move_group.stop()

"""
#We can plan a motion for this group to a desired pose for the end-effector:
pose_goal = geometry_msgs.msg.Pose()
#Position
pose_goal.position.x = 0.11
pose_goal.position.y = -1.91442517679e-05
pose_goal.position.z = 0.675
#Orientation
pose_goal.orientation.x = -0.000567874271249
pose_goal.orientation.y = -0.000224426230725
pose_goal.orientation.z = 0.37552177936
pose_goal.orientation.w = 0.926813368687

move_group.set_pose_target(pose_goal)

plan = move_group.go(wait=True)
# Calling `stop()` ensures that there is no residual movement
move_group.stop()
# It is always good to clear your targets after planning with poses.
# Note: there is no equivalent function for clear_joint_value_targets()
move_group.clear_pose_targets()


