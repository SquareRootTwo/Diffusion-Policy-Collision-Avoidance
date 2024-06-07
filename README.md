# Diffusion Policy for Collision Avoidance in a Two-Arm Robot Setup

Thesis link: [https://doi.org/10.3929/ethz-b-000675013](https://doi.org/10.3929/ethz-b-000675013)

## Abstract
Diffusion models in robotics have shown great potential with a wide range of applicability. Especially their capability to model multi-modal data distribution has many benefits when learning a task such as collision-free trajectory optimisation with dynamic moving objects. Collision avoidance is a central problem in robotics where classical approaches still suffer from non-optimal solutions or high computational costs. 

In this work, we present four diffusion models, capable of predicting collision-free trajectories in a pick and place setup of two moving robot arms. The best model achieves a success rate of 88.1% in producing collision-free trajectories, while the worst one succeeds in 76.2% of the episodes. Furthermore, we analyse the models in terms of their accuracy in reaching the target pose, their capability of predicting smooth trajectories, and their success rate in generating collision-free trajectories.

## Results Simulations

The following section shows simulations for all four models.

### Cartesian Space U-Net
[https://youtu.be/iFDBXmPkvQ8](https://youtu.be/iFDBXmPkvQ8)

### Joint Space U-Net
[https://youtu.be/elthZgPWOlc](https://youtu.be/elthZgPWOlc)

### Cartesian Space Transformer
[https://youtu.be/O_Okx1t4Tik](https://youtu.be/O_Okx1t4Tik)

### Joint Space Transformer
[https://youtu.be/97LsCvt__JM](https://youtu.be/97LsCvt__JM)


# Code
## Dataset Generation

To run the dataset generation docker conatiner, make sure to have the latest curobo docker container on your system (click [here](https://curobo.org/get_started/5_docker_development.html) for the curobo docker setup, [here](https://github.com/NVlabs/curobo/blob/main/docker/x86.dockerfile) for the curobo dockerfile and [here](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html) for the isaac sim docker setup).


### Ghost Robot Dataset Generation
To generate the `Ghost Robot Dataset` run the following line (change the docker run command in the shell script if docker should not run with root privileges)


```
cd src && sh ./generate_ghost_robot_dataset.sh
```

### Pick and Place Dataset Generation

Once the `Ghost Robot Dataset` is created, verify that the file `src/data/curobo_panda_pick_and_place_ghost_robot_dataset.parquet` exists and then run

```
sh ./generate_pick_and_place_dataset.sh
```

from the `src/` directory

This will run the docker contianer with the script for the second dataset. The file will be stored under `./src/data/curobo_panda_pick_and_place_robot_collision_dataset.parquet`.

Note: it takes a lot of time to sample both datasets and there are a lot of inefficiencies in the sampling process that could be improved. To make the sampling more efficient one could first remove the simulation (e.g. use cuRobo without isaac sim) and then use batched computation of trajectories. 


To replay the `Pick and Place Dataset` run 

```
python3 eval/replay_dataset.py
```

## Training the Models

First make sure that the `Pick and Place Dataset` is either generated or downloaded from the following link: [here](https://polybox.ethz.ch/index.php/s/nRt6i0ZX1BnDgIx/download) and placed under `/src/data/curobo_panda_pick_and_place_robot_collision_dataset.parquet`.

To download the dataset run

```
cd src/ && mkdir data/ && cd data/ && wget -O curobo_panda_pick_and_place_robot_collision_dataset.parquet https://polybox.ethz.ch/index.php/s/nRt6i0ZX1BnDgIx/download
```

Then add your wandb key to the `src/cfg/wandb.yaml` file for the logging of the train statistics.
To train one of the four models, navigate to the `src/` folder and run one of the following lines:

```
python3 train/train_diffusion_policy_cartesian_space_unet.py
```

```
python3 train/train_diffusion_policy_joint_space_unet.py
```

```
python3 train/train_diffusion_policy_cartesian_space_transformer.py
```

```
python3 train/train_diffusion_policy_joint_space_transformer.py
```

## Simulating the Models


First provide the correct timestamp from the training run together with the `model_type` ("unet" or "transformer") in `EvalConfig` class. Then run the script

```
python3 eval/replay_cartesian_space_model.py
```

for the Cartesian space models and for the joint space models run 

```
python3 eval/replay_joint_space_model.py
```

## Plotting the Results from the Simulations

In order to generate the plots used for the thesis use the scripts in the folder `src/plt_scripts/`. Note that the timestamps need to be changed for the plotting scripts to work with newly trained models.

## Acknowledgement

The code in the following srcipts are adapted from the Diffusion Policy paper ([link](https://github.com/real-stanford/diffusion_policy?tab=readme-ov-file)). 

Train files:
- src/train/train_diffusion_policy_cartesian_space_transformer.py
- src/train/train_diffusion_policy_cartesian_space_unet.py
- src/train/train_diffusion_policy_joint_space_transformer.py
- src/train/train_diffusion_policy_joint_space_unet.py

Neural Network Model files:
- src/models/diffusion_transformer.py
- src/models/unet_diffusion_policy.py

The dataset generation scripts were inspired and adapted from the official Motion Gen cuRobo tutorials from their Github page ([link](https://github.com/NVlabs/curobo/blob/main/examples/isaac_sim/motion_gen_reacher.py)). 