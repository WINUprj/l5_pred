cmake_minimum_required(VERSION 2.8.3)
project(digit_recognition)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  img_msgs
  duckietown_msgs
  roscpp
  rospy
)


catkin_package()
