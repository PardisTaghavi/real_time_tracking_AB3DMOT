# Real_Time_Tracking_AB3DMOT
Modification of "AB3DMOT" code for real-time tracking in ROS
----------------------------------------------------------------------
The Code will be published before 30th of June.



- clone the repo:
```git clone https://github.com/PardisTaghavi/real_time_tracking_AB3DMOT.git ```

- create conda environments with the required libraries:
```conda env create --file environment.yml```
- activate conda environment: ``` conda activate AB3D```
- you should build the package in a catkin workspace ``` catkin build ab3d or catkin_make```
- Don't forget to source the package ```source devel/setup.bash ```
- ```chmod +x modelROS.py``` (do it one time to change accessibility of the file)

a rosbag have been created from kitti dataset. you can save the file and play the rosbag on a seperate terminal 
``` rosbag play *.bag```

- launch the conde: ``` roslaunch ab3d ab3dLaunch.launch ```

