
### "AB3DMOT" modification for real-time tracking in ROS

### Requirements
Ubuntu 18.04/ ROS Melodic

### Installation

- clone this repo in catkin_ws/src directory:

```git clone https://github.com/PardisTaghavi/real_time_tracking_AB3DMOT.git ```
``` cd ab3d/src ```

- create conda environments with the required libraries:

```conda env create --file environment.yml```
- activate conda environment: 

```conda activate AB3D```
- install AB3DMOT and mmdetection3d repositories and follow their instructions for installatoion 

```git clone https://github.com/xinshuoweng/AB3DMOT.git```
```git clone https://github.com/open-mmlab/mmdetection3d.git```


- Build the package in catkin workspace with

``` catkin build ab3d or catkin_make```
and Don't forget to source the package ```source devel/setup.bash ```

- ```chmod +x modelROS.py``` (do it one time to change accessibility of the file)

a rosbag file from kitti dataset is available in the following link for testing purposes: https://drive.google.com/file/d/1WHaCPYf0tYoGBJDB780DLfgqdfXo0V_p/view?usp=sharing
you should play the rosbag on a seperate terminal

``` rosbag play <name of rosbag>.bag```


- launch the code: 

``` roslaunch ab3d ab3dLaunch.launch ```


![ezgif com-gif-maker](https://github.com/PardisTaghavi/real_time_tracking_AB3DMOT/blob/main/TrackingDemo.gif)

-------------------------------------------------------------------
#### "Special thanks to the authors of the original code AB3DMOT and mmdetection3d for laying the foundation of this project."
AB3DMOT : https://github.com/xinshuoweng/AB3DMOT

mmdetection3d : https://github.com/open-mmlab/mmdetection3d

kitti dataset : https://www.cvlibs.net/datasets/kitti/
