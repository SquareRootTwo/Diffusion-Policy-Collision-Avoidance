from dataclasses import dataclass
from dataclasses import asdict
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from scipy.spatial.transform import Rotation as R

# logging with wandb
import datetime
import wandb
import time

# add root to path
import sys, yaml
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

# custom imports
from src.models.diffusion_transformer import TransformerForDiffusion
from src.dataset.dataset_loader import Panda_Diffusion_Policy_Full_Trajectory_Dataset_Loader
from src.utils.normalisation import normalize_data, unnormalize_data
from src.utils.eval_statistics import compute_loss_fk

# set seed for reproducibility
torch.manual_seed(1337)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# open yaml file for wandb key
with open(os.path.join(root_path, os.path.join(root_path, "src/cfg/wandb.yaml")), 'r') as stream:
    try:
        wandb_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

@dataclass
class TrainConfig:
    pred_horizon: int = 34
    obs_horizon: int = 34
    # action_horizon: int = 8
    augmented_df_path: str = os.path.join(root_path, "src/data/curobo_panda_pick_and_place_robot_collision_dataset_augmented.parquet")
    normal_df_path: str = os.path.join(root_path, "src/data/curobo_panda_pick_and_place_robot_collision_dataset.parquet")

    num_workers: int = 6
    train_fraction: float = 0.8

    # dataset parameters
    load_dataset_to_memory: bool = True
    augment_data: bool = False

    # nn parameters
    n_layer: int = 12 # TODO: test 48
    n_head: int = 12
    n_emb: int = 768
    causal_attn: bool = True
    n_cond_layers: int = 0 # 12

    # training parameters
    num_diffusion_iters: int = 100 
    batch_size: int = 256
    num_epochs: int = 1500
    lr: float = 1e-4
    weight_decay: float = 1e-6

    action_dim: int = 7
    conditional_dim: int = 509
    target_dim: int = 7

    # wandb config file
    wandb_offline: bool = False
    wandb_logging: bool = True
    wandb_project: str = 'MARS'
    wandb_group: str = 'Diffusion_Policy_Joint_Space_Transformer'
    wandb_name: str = f'Diffusion_Policy_Transformer_{timestamp}'
    wandb_entity: str = 'spieler'
    wandb_key: str = wandb_config.get('wandb_key')

    # logging parameters
    wandb_model_save_freq: int = 200

    # evaluation
    eval_step_interval: int = 20
    run_eval : bool = True


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = TrainConfig()
    # parameters
    # Obs:      [] -> context
    # Action:   [] -> state
    #|o|o|o|o|o|o|o|o|o|o|o|o|o|o|o|o| observations: 34
    #|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a| actions executed: 34
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 34

    if config.augment_data:
        dataset = Panda_Diffusion_Policy_Full_Trajectory_Dataset_Loader(
            df_path = config.augmented_df_path,
            fraction_to_use=.3,
            eval=False,
            augment_data=False,
            load_dataset_to_memory=config.load_dataset_to_memory,
        )
    else:
        dataset = Panda_Diffusion_Policy_Full_Trajectory_Dataset_Loader(
            df_path = config.normal_df_path,
            fraction_to_use=config.train_fraction,
            eval=False,
            augment_data=False,
            load_dataset_to_memory=config.load_dataset_to_memory,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )

    if config.run_eval:
        eval_dataset = Panda_Diffusion_Policy_Full_Trajectory_Dataset_Loader(
            df_path = config.normal_df_path,
            fraction_to_use=0.9,
            eval=True,
            augment_data=False,
            load_dataset_to_memory=config.load_dataset_to_memory
        )

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
        )

    # For data normalization/unnormalization
    obs_stats = torch.from_numpy(
        np.genfromtxt(os.path.join(
            root_path, 
            'src/data/normalisation_obs.csv'
        ), delimiter=',')
    ).to(torch.float32).to(device=device)

    action_stats = torch.from_numpy(
        np.genfromtxt(os.path.join(
            root_path, 
            'src/data/normalisation_action.csv'
        ), delimiter=',')
    ).to(torch.float32).to(device=device)

    # create NN model
    noise_pred_net = TransformerForDiffusion(
        input_dim=config.action_dim,
        output_dim=config.action_dim,
        horizon=config.pred_horizon,
        cond_dim=config.conditional_dim,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_emb = config.n_emb,
        causal_attn=config.causal_attn,
        time_as_cond=True,
        obs_as_cond=True,
        n_cond_layers=config.n_cond_layers
    )

    # print number of model parameters
    num_params = sum(p.numel() for p in noise_pred_net.parameters())
    print(f"Number of model parameters: {num_params / 1e6} M")


    # wandb logging
    if config.wandb_logging:
        wandb.login(key=config.wandb_key)
        if config.wandb_offline:
            wandb.init(
                project=config.wandb_project,
                group=config.wandb_group,
                name=config.wandb_name,
                entity=config.wandb_entity,
                mode='offline',
            )
        else:
            wandb.init(
                project=config.wandb_project,
                group=config.wandb_group,
                name=config.wandb_name,
                entity=config.wandb_entity,
            )

        # log config
        config_dict = asdict(config)

        # remove sensitive information
        config_dict.pop('wandb_key', None)
        
        # add parameters to config dict
        config_dict['num_params'] = num_params

        # log config to wandb
        wandb.config.update(config_dict)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    noise_pred_net.to(device)

    # Exponential Moving Average Model
    ema_model = torch.optim.swa_utils.AveragedModel(
        noise_pred_net,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
    )

    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=config.lr, 
        weight_decay=config.weight_decay)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * config.num_epochs
    )

    os.makedirs(os.path.join(root_path, f"src/logs/diffusion_policy/transformer/{timestamp}_run"), exist_ok=True)
    
    with tqdm(range(config.num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    nobs = nbatch['obs'].to(device)
                    naction = nbatch['action'].to(device)

                    # normalize data
                    nobs = normalize_data(nobs, obs_stats)
                    naction = normalize_data(naction, action_stats)
                    naction = naction[:, :, :7] # only use joint states -> fk only model

                    B = nobs.shape[0]

                    obs_cond = nobs

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # forward diffusion process/adding noise at randomly sampled timesteps
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, cond=obs_cond
                    )

                    # loss
                    loss_mse = nn.functional.mse_loss(noise_pred, noise) 
                    # total loss
                    loss = loss_mse

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update ema model weights
                    ema_model.update_parameters(noise_pred_net)

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

                    # wandb logging
                    if config.wandb_logging:
                        wandb.log({
                            'train/loss': loss_cpu, 
                            'train/loss_mse': loss_mse.item(),
                            'train/lr': lr_scheduler.get_last_lr()[0]
                        })


            if (epoch_idx+1) % config.eval_step_interval == 0 and config.run_eval:
                noise_pred_net.eval()

                with torch.no_grad():
                    eval_loss = list()
                    eval_loss_last_joint = list()
                    eval_loss_first_joint = list()
                    eval_loss_jerk = list()
                    eval_loss_acc = list()
                    eval_loss_vel = list()
                    eval_loss_last_position = list()
                    eval_loss_last_angle = list()

                    for eval_i, eval_batch in tqdm(enumerate(eval_dataloader), desc="Eval"):
                        # device transfer
                        nobs = eval_batch['obs'].to(device)
                        naction = eval_batch['action'].to(device)

                        nobs = normalize_data(nobs, obs_stats)
                        naction = normalize_data(naction, action_stats)
                        naction = naction[:, :, :7] # only use joint states -> fk only model

                        B = nobs.shape[0]
                        
                        obs_cond = nobs
                        # sample noise to add to actions
                        noise = torch.randn(naction.shape, device=device)

                        # sample a diffusion iteration for each data point
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps,
                            (B,), device=device
                        ).long()

                        # add noise to the clean images according to the noise magnitude at each diffusion iteration
                        # (this is the forward diffusion process)
                        noisy_actions = noise_scheduler.add_noise(
                            naction, noise, timesteps
                        )

                        # predict the noise residual
                        noise_pred = noise_pred_net(
                            noisy_actions, timesteps, cond=obs_cond
                        )

                        # loss
                        # L2 loss
                        loss_mse = nn.functional.mse_loss(noise_pred, noise) 
                        eval_loss.append(loss_mse.item())

                        # sample noise to add to actions
                        noisy_actions = torch.randn(naction.shape, device=device)

                        for k in noise_scheduler.timesteps:
                            timesteps = torch.ones(B, device=device, dtype=torch.long) * k

                            noise_pred = noise_pred_net(
                                noisy_actions, 
                                timestep=k,
                                cond=obs_cond
                            )

                            noisy_actions = noise_scheduler.step(
                                model_output=noise_pred,
                                timestep=k,
                                sample=noisy_actions
                            ).prev_sample
    
                        # zero pad action_pred for unnormalization
                        noisy_actions = torch.cat([
                            noisy_actions, torch.zeros_like(noisy_actions, device=device)], 
                            dim=2
                        )
                        action_pred = unnormalize_data(noisy_actions, action_stats)
                        # zero pad action for unnormalization
                        naction = torch.cat([
                            naction, torch.zeros_like(naction, device=device)], 
                            dim=2
                        )
                        action_gt = unnormalize_data(naction, action_stats)

                        loss_dict = compute_loss_fk(action_pred, action_gt, (eval_batch["target"][:, -1, :]).to(device))

                        eval_loss_last_joint.append(loss_dict["loss_final_joint_state"].item())
                        eval_loss_first_joint.append(loss_dict["loss_start_joint_state"].item())

                        eval_loss_last_position.append(loss_dict["loss_final_position"].item())
                        eval_loss_last_angle.append(loss_dict["loss_angle_final"].item())

                        eval_loss_jerk.append(loss_dict["loss_jerk"].item())
                        eval_loss_acc.append(loss_dict["loss_acc"].item())
                        eval_loss_vel.append(loss_dict["loss_vel"].item())


                    eval_loss_mean = np.mean(eval_loss)
                    eval_loss_std = np.std(eval_loss)

                    eval_loss_first_joint_mean = np.mean(eval_loss_first_joint)
                    eval_loss_first_joint_std = np.std(eval_loss_first_joint)

                    eval_loss_last_joint_mean = np.mean(eval_loss_last_joint)
                    eval_loss_last_joint_std = np.std(eval_loss_last_joint)

                    eval_loss_jerk_mean = np.mean(eval_loss_jerk)
                    eval_loss_jerk_std = np.std(eval_loss_jerk)

                    eval_loss_acc_mean = np.mean(eval_loss_acc)
                    eval_loss_acc_std = np.std(eval_loss_acc)

                    eval_loss_vel_mean = np.mean(eval_loss_vel)
                    eval_loss_vel_std = np.std(eval_loss_vel)
                    
                    eval_loss_last_position_mean = np.mean(eval_loss_last_position)
                    eval_loss_last_position_std = np.std(eval_loss_last_position)

                    eval_loss_last_angle_mean = np.mean(eval_loss_last_angle)
                    eval_loss_last_angle_std = np.std(eval_loss_last_angle)

                    if config.wandb_logging:
                        wandb.log({
                            'eval_mean/loss_mse_mean': eval_loss_mean, 
                            'eval_std/loss_mse_std': eval_loss_std,
                            'eval_mean/loss_first_joint_mean': eval_loss_first_joint_mean,
                            'eval_std/loss_first_joint_std': eval_loss_first_joint_std,
                            'eval_mean/loss_last_joint_mean': eval_loss_last_joint_mean,
                            'eval_std/loss_last_joint_std': eval_loss_last_joint_std,
                            'eval_mean/loss_jerk_mean': eval_loss_jerk_mean,
                            'eval_std/loss_jerk_std': eval_loss_jerk_std,
                            'eval_mean/loss_acc_mean': eval_loss_acc_mean,
                            'eval_std/loss_acc_std': eval_loss_acc_std,
                            'eval_mean/loss_vel_mean': eval_loss_vel_mean,
                            'eval_std/loss_vel_std': eval_loss_vel_std,
                            'eval_mean/loss_last_position_mean': eval_loss_last_position_mean,
                            'eval_std/loss_last_position_std': eval_loss_last_position_std,
                            'eval_mean/loss_last_angle_mean': eval_loss_last_angle_mean,
                            'eval_std/loss_last_angle_std': eval_loss_last_angle_std
                        })

                    noise_pred_net.train()

            # save model every wandb_model_save_freq epochs
            if (epoch_idx+1) % config.wandb_model_save_freq == 0:
                # save model
                model_path = f"src/logs/diffusion_policy/transformer/{timestamp}_run/{timestamp}_epoch_{epoch_idx}_weights.pt"
                model_path = os.path.join(root_path, model_path)
                torch.save(noise_pred_net.state_dict(), model_path)

                # save ema model
                ema_model_path = f"src/logs/diffusion_policy/transformer/{timestamp}_run/{timestamp}_ema_epoch_{epoch_idx}_weights.pt"
                ema_model_path = os.path.join(root_path, ema_model_path)
                torch.save(ema_model.state_dict(), ema_model_path)

                # store checkpoint
                torch.save({
                    'epoch': epoch_idx,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'epoch_loss': epoch_loss,
                    'eval_loss': eval_loss
                }, os.path.join(root_path, f"src/logs/diffusion_policy/transformer/{timestamp}_run/{timestamp}_checkpoint.pt"))


            tglobal.set_postfix(loss=np.mean(epoch_loss))

            if config.wandb_logging:
                wandb.log({'train/epoch_loss': np.mean(epoch_loss)})


    # save final ema model
    ema_model_path = f"src/logs/diffusion_policy/transformer/{timestamp}_run/{timestamp}_ema_weights.pt"
    ema_model_path = os.path.join(root_path, ema_model_path)

    # save ema model
    print(f"Saving EMA model to {ema_model_path}")
    torch.save(ema_model.state_dict(), ema_model_path)

    # save final model
    model_path = f"src/logs/diffusion_policy/transformer/{timestamp}_run/{timestamp}_final_weights.pt"
    model_path = os.path.join(root_path, model_path)
    torch.save(noise_pred_net.state_dict(), model_path)


if __name__ == "__main__":
    train()