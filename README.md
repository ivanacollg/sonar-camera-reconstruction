# sonar-camera-reconstruction

This repo contains the code derived from the paper "Opti-Acoustic Scene Reconstruction in Highly Turbid Underwater Environments" (2025), which presents an imaging sonar and monocular camera merging system for scene reconstruction. 

# Dependencies
This codebase is ROS native and will require a ROS installation. It can be used without ROS, but will require some work.

    - ROS Noetic
    - Python3

Presently, this codebase uses our sonar image system in Argonaut
```
    git clone https://github.com/jake3991/Argonaut.git
```
If recontruction from multiple poses is desired, an odometry source is requiered, we use our sonar-SLAM framework for this:
```
    git clone https://github.com/jake3991/sonar-SLAM.git
```

# Set Up
```
    mkdir -p catkin_ws
    cd catkin_ws
    git clone https://github.com/jake3991/Argonaut.git
    git clone https://github.com/jake3991/sonar-SLAM.git
    git clone https://github.com/ivanacollg/sonar-camera-reconstruction.git
```
Follow instruction for Argonaut setup as well. 

# Running Code
```
    roslaunch sonar_camera_merge merge.launch
```

# Use Guide
Input Topics:
- Camera image topic
- Sonar topic
- Robot Odometry topic 

Output Topics:
- Segmented Image topic
- Sonar features Image topic
- Point Cloud reconstruction

Parameters:
- Monocular Camera parameters
- Sonar parameters
- Merge parameters

# Sample data
Download sample data:
https://drive.google.com/file/d/1IXFs3ATa16V4y8qqJ_ss_8yxvXu2n6vx/view?usp=drive_link

# Citations
If you use this repo please cite the following work. 
```
@inproceedings{
    title={Opti-Acoustic Scene Reconstruction in Highly Turbid Underwater Environments},
    author={Ivana Collado-Gonzalez, John McConnell, Paul Szenher, and Brendan Englot},
    booktitle={},
    year={2025},
    organization={}
}
```