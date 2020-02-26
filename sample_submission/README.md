# NuScenes Prediction Challenge resources

docker/ directory contains docker image required to run NuScenes Prediction Challenge models

## Requirements

- machine with GPU card and nvidia drivers (for GPU support)
- nvidia-docker https://github.com/NVIDIA/nvidia-docker. You can use generic docker image if you don't need GPU support
- nuScenes dataset unpacked
- cloned nuScenes-devkit repo https://github.com/nutonomy/nuscenes-devkit

## Usage
- pull docker image
```
docker pull nuscenes/dev-challenge:latest
```
- create directory for output data
```
mkdir mkdir -p ~/Documents/submissions
```
- have your sources available
- run docker container
```
cd <NUSCENES ROOT DIR>
docker run [ --gpus all ] -ti --rm \
   -v <PATH TO DATASET>:/data/sets/nuscenes \
   -v <PATH TO nuScenes-devkit ROOT DIR>/python-sdk:/nuscenes-dev/python-sdk \
   -v <PATH TO YOUR SOURCES>:/nuscenes-dev/predict \
   -v ~/Documents:/nuscenes-dev/Documents \
   nuscenes/dev-challenge:latest
```
