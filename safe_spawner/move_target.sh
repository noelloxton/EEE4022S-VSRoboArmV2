#! /bin/bash

#TO INSTAL x-term emulator run
#sudo apt-get update -y
#sudo apt-get install -y x-terminal-emulator

#edit -x -y -z for location of target in sim -Y for yaw/rotation
x-terminal-emulator -e rostopic pub -r 20 /gazebo/set_model_state gazebo_msgs/ModelState '{model_name: target, pose: { position: { x: 2.2, y: 0.2, z: 1.2 }, orientation: {x: 0, y: 0, z: 0, w: 1 } }, twist: { linear: { x: 0, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0}  }, reference_frame: world }'

