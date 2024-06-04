import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from dataclasses import dataclass
from typing import Sequence
import os, sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
from time import time
from tqdm import tqdm

# import diffusion model
from src.models.diffusion_transformer import TransformerForDiffusion
from src.models.unet_diffusion_policy import ConditionalUnet1D
from src.utils.normalisation import normalize_data, unnormalize_data
    
"""
This script evaluates the time taken to predict a single timestep of the diffusion model.
To change the model (unet or transformer), change the model_type in the TrainConfig class.
"""

@dataclass
class TrainConfig:
    pred_horizon: int = 34
    obs_horizon: int = 34
    action_horizon: int = 8

    num_workers: int = 5
    train_fraction: float = 0.8
    dtype = torch.float32

    # dataset parameters
    augment_data: bool = False
    dtype = torch.float32

    # model_type = "unet" 
    model_type = "transformer"

    # Transformer nn parameters
    n_layer: int = 12
    n_head: int = 12
    n_emb: int = 768
    causal_attn: bool = True 
    n_cond_layers: int = 0

    # Unet nn parameters
    down_dims: Sequence[int] = (256, 512, 1024, 1024) 
    kernel_size: int = 3
    diffusion_step_embed_dim: int = 256
    n_groups: int = 8
    num_diffusion_iters: int = 100
    num_eval_iters: int = 10
    conditional_dim: int = 509
    action_dim: int = 7

    # training parameters
    batch_size: int = 256
    num_epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-6
    jerk_loss_alpha: float = 0.2
    acc_loss_alpha: float = 0.2
    vel_loss_alpha: float = 0.2

    target_dim: int = 7

    # logging parameters
    wandb_model_log_freq: int = 30
    wandb_model_save_freq: int = 100

    # evaluation
    eval_step_interval: int = 100


def main():
    config = TrainConfig()

    run_timestamp = "2024-04-22_00-04-23" # transformer
    # run_timestamp = "2024-04-18_00-49-51"

    device = "cuda"
    obs_norm_path = os.path.join(root_path, 'src/data/normalisation_obs.csv')
    action_norm_path = os.path.join(root_path, 'src/data/normalisation_action.csv')
    
    action_stats = torch.from_numpy(
        np.genfromtxt(action_norm_path, delimiter=',')
    ).to(device=device)

    obs_stats = torch.from_numpy(
        np.genfromtxt(obs_norm_path, delimiter=',')
    ).to(device=device)

    ema_model_path = os.path.join(root_path, f"src/logs/diffusion_policy/{config.model_type}/{run_timestamp}_run", f"{run_timestamp}_ema_weights.pt")

    if config.model_type == "transformer":
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
    else:
        noise_pred_net = ConditionalUnet1D(
            input_dim=config.action_dim,
            cond_dim=config.conditional_dim*config.obs_horizon,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            n_groups=config.n_groups,
        )

    noise_pred_net = torch.optim.swa_utils.AveragedModel(
        noise_pred_net,
    )

    noise_pred_net.load_state_dict(torch.load(ema_model_path))

    noise_pred_net = noise_pred_net.to(device)
    noise_pred_net.eval()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    noise_scheduler.set_timesteps(config.num_eval_iters, device=device)


    time_diff = []

    # replan trajectory
    with torch.no_grad():
        for i in tqdm(range(100)):
            t1 = time()
            obs = torch.rand((1,34,509)).to("cuda")
            nobs = normalize_data(obs, obs_stats)
            # nobs = nobs.flatten(start_dim=1).to(torch.float32)

            obs_cond = nobs.to(dtype=torch.float32)
            B = 1

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, 34, 7), device=device)
            naction = noisy_action.to(dtype=torch.float32)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = noise_pred_net(
                    sample=naction,
                    timestep=k,
                    cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            naction = torch.cat([naction, torch.zeros_like(naction, device=device)], dim=2)
            action_pred = unnormalize_data(naction, stats=action_stats).squeeze(0)

        t2 = time()

        time_diff.append(t2 - t1)

    print(f"avg prediction time: {np.mean(time_diff)} s")


if __name__ == "__main__":
    main()