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

NOTE: The docker image uses 1000:1000 uid:gid
If this is different from your local setup, you may want to add this options into `docker run` command
```
--user `id -u`:`id -g` -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group
```

- execute your script inside docker container
```
source activate /home/nuscenes/.conda/envs/nuscenes

python submission/do_inference.py --version v1.0-mini --data_root /data/sets/nuscenes --split_name mini_val --model_weights 'foo' --output_dir /nuscenes-dev/Documents/submissions --submission_name sample_23
```
