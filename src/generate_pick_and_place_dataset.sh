# build docker image
sudo docker build -f curobo.dockerfile -t custom_curobo .

# run docker image to generate dataset
sudo docker run --name container --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
  --privileged \
  -v $HOME/.Xauthority:/root/.Xauthority \
  -e DISPLAY \
  -v /usr/share/vulkan/icd.d/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  -v ./data:/root/dataset:rw  \
  -v ./dataset_generation/ik_pick_and_place_robot_collision.py:/src/main.py \
  --volume /dev:/dev \
  custom_curobo:latest
  # -v ./dataset_generation/ik_pick_and_place_collision.py:/src/main.py \
