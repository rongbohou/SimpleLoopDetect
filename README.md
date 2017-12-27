## Prerequisites
### boost
```
sudo apt-get install libboost-dev libboost-filesystem-dev
```
### Eigen3
```
sudo apt-get install libeigen3
```

### DBoW2
Install from [dorian3d/DBoW2](https://github.com/dorian3d/DBoW2)

## Usage
```
rosrun LoopDetector myloopdetector path_to_vocabulary paht_to_BRIEF_PATTERN_FILE
```
For example:
```
rosrun LoopDetector myloopdetector /home/bobo/catkin_ws/src/LoopDetector/build/resources/brief_k10L6.voc.gz /home/bobo/catkin_ws/src/LoopDetector/build/resources/brief_pattern.yml
```
OR
```
cd build
./myloopdetector /home/bobo/catkin_ws/src/LoopDetector/build/resources/brief_k10L6.voc.gz /home/bobo/catkin_ws/src/LoopDetector/build/resources/brief_pattern.yml
```
