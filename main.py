import os
import json
from pathlib import Path
from itertools import cycle
from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import trange
from torch.utils.data import DataLoader

from models import ncsnpp
import losses
import sde_lib
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from utils_train import preprocess
from config import Configuration

def restore_checkpoint(ckpt_dir, state, device):
    if not Path(ckpt_dir).exists():
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state
    
def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)

def train():

    config = Configuration.parse_args()

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataname = config.dataname
    config.data_dataset = dataname
    dataset_dir = f'data/{dataname}'

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']
    num_numericals = len(info["num_col_idx"])
    dataset, _ = preprocess(dataset_dir, task_type=task_type, cat_encoding='one-hot', use_resbit=config.data_use_resbit)
    train_z = torch.tensor(dataset.X_num['train'])
    config.data_image_size = train_z.shape[1]
    train_data = train_z
    
    if config.data_scale_ohe:
        train_data[:, num_numericals:] = train_data[:, num_numericals:] * 2 - 1

    train_ds_loader = cycle(
        DataLoader(
            train_data,
            batch_size=config.training_batch_size,
            shuffle=True,
            num_workers=4
        )
    )

    # Initialize model.
    score_model: torch.nn.Module = mutils.create_model(config)
    print(score_model)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model_ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    if config.optim_optimizer == 'RAdamScheduleFree':
        optimizer.train()
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    initial_step = int(state['step'])

    # Setup SDEs
    sde = sde_lib.RectifiedFlow(
        init_type=config.sampling_init_type,
        noise_scale=config.sampling_init_noise_scale,
        use_ode_sampler=config.sampling_use_ode_sampler,
        num_numericals=num_numericals,
        scale_ohe=config.data_scale_ohe
    )

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    reduce_mean = config.training_reduce_mean
    likelihood_weighting = config.training_likelihood_weighting
    train_step_fn = losses.get_step_fn(
        sde, train=True, optimize_fn=optimize_fn, 
        reduce_mean=reduce_mean, 
        likelihood_weighting=likelihood_weighting
    )

    num_train_steps = config.training_n_iters
    exp_setting = config.exp

    print(f"Starting training loop at step {initial_step}.")
    loss_history = []
    pbar = trange(initial_step, num_train_steps)
    for i in pbar:
        data: torch.Tensor = next(train_ds_loader)
        batch = data.to(config.device, torch.float32)
        loss: torch.Tensor = train_step_fn(state, batch)

        pbar.set_postfix(
            OrderedDict(
                loss=loss.item()
            )
        )
        loss_history.append({"loss": loss.item(), "iter": i})

        if i != 0 and i % config.training_snapshot_freq == 0 or i == num_train_steps - 1:
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{exp_setting}_{i:06d}.pth'), state)
        
        if i == 29999:
            sde.switch_loss_to_hurber = True
            train_step_fn = losses.get_step_fn(
                sde, train=True, optimize_fn=optimize_fn, 
                reduce_mean=reduce_mean, 
                likelihood_weighting=likelihood_weighting
            )


    df = pd.DataFrame(loss_history)
    df.to_csv(f"loss_log_{exp_setting}.csv", index=False)
    x = df["iter"].values
    y = df["loss"].values
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    plt.xlabel('iteration')
    plt.plot(x, y, label='train_loss')
    plt.legend()

    plt.savefig(f"loss_log_{exp_setting}.png")


if __name__ == "__main__":
    train()
