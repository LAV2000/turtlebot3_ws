sudo apt install python3-vcstool

sudo apt install ros-humble-gazebo-*
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup
sudo apt install ros-galactic-slam-toolbox

---Cai dat goi turtlebot3--
vcs import . < turtlebot3.ros2.repos

---export model. Co the ghi them vao file .bashrc---
export GAZEBO_MODELPATH=$GAZEBO_MODEL_PATH:~/turtlebot3_ws/src/turtlebot3/turtlebot3_simulations/turtlebot3_gazebo/models
export TURTLEBOT3_MODEL=waffle