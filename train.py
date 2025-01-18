import time
import gymnasium
import argparse
from tensorboardX import SummaryWriter
import cv2
import numpy as np
import warnings
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import copy
import colorama
import random
import json
import shutil
import pickle
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from utils import seed_np_torch, Logger, load_config
from replay_buffer import ReplayBuffer
import env_wrapper
import agents
from sub_models.functions_losses import symexp
from sub_models.world_models import WorldModel, MSELoss


def build_single_env(env_name, image_size, seed):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.SeedEnvWrapper(env, seed=seed)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfo(env)
    return env


def build_vec_env(env_name, image_size, num_envs, seed):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size, seed)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def train_world_model_step(replay_buffer: ReplayBuffer, world_model: WorldModel, batch_size, demonstration_batch_size, batch_length, logger):
    obs, action, reward, termination = replay_buffer.sample(batch_size, demonstration_batch_size, batch_length)
    world_model.update(obs, action, reward, termination, logger=logger)


@torch.no_grad()
def world_model_imagine_data(replay_buffer: ReplayBuffer,
                             world_model: WorldModel, agent: agents.ActorCriticAgent,
                             imagine_batch_size, imagine_demonstration_batch_size,
                             imagine_context_length, imagine_batch_length,
                             log_video, logger):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    agent.eval()

    sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
        imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length)
    latent, action, reward_hat, termination_hat = world_model.imagine_data(
        agent, sample_obs, sample_action,
        imagine_batch_size=imagine_batch_size+imagine_demonstration_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        logger=logger
    )
    return latent, action, None, None, reward_hat, termination_hat


def joint_train_world_model_agent(env_name, max_steps, num_envs, image_size,
                                  replay_buffer: ReplayBuffer,
                                  world_model: WorldModel, agent: agents.ActorCriticAgent,
                                  train_dynamics_every_steps, train_agent_every_steps,
                                  batch_size, demonstration_batch_size, batch_length,
                                  imagine_batch_size, imagine_demonstration_batch_size,
                                  imagine_context_length, imagine_batch_length,
                                  save_every_steps, seed, logger, name, device:torch.device):
    # create ckpt dir
    os.makedirs(f"ckpt/{name}", exist_ok=True)

    # build vec env, not useful in the Atari100k setting
    # but when the max_steps is large, you can use parallel envs to speed up
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs, seed=seed)
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)

    # reset envs and variables
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    # sample and train
    for total_steps in tqdm(range(max_steps//num_envs)):
        # sample part >>>
        if replay_buffer.ready():
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0:
                    action = vec_env.action_space.sample()
                else:
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = torch.Tensor(model_context_action).to(device)
                    prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False
                    )

            context_obs.append(rearrange(torch.Tensor(current_obs).to(device), "B H W C -> B 1 C H W")/255)
            context_action.append(action)
        else:
            action = vec_env.action_space.sample()

        obs, reward, done, truncated, info = vec_env.step(action)
        replay_buffer.append(current_obs, action, reward, np.logical_or(done, info["life_loss"]))
        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag[i]:
                    logger.log(f"sample/{env_name}_reward", sum_reward[i])
                    logger.log(f"sample/{env_name}_episode_steps", current_info["episode_frame_number"][i]//4)  # framskip=4
                    logger.log("replay_buffer/length", len(replay_buffer))
                    episode_dict = {
                        f"sample/{env_name}_episode_reward": sum_reward[i],
                        f"sample/{env_name}_episode_steps": current_info["episode_frame_number"][i]//4,
                        "replay_buffer/length": len(replay_buffer)
                    }
                    wandb.log(episode_dict)
                    sum_reward[i] = 0

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        # <<< sample part
        # train world model part >>>
        if replay_buffer.ready() and total_steps % (train_dynamics_every_steps//num_envs) == 0:
            start_time = time.time()
            train_world_model_step(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=batch_size,
                demonstration_batch_size=demonstration_batch_size,
                batch_length=batch_length,
                logger=logger
            )

            wandb.log({'runtime/world_model': time.time()-start_time})
        # <<< train world model part

        # train agent part >>>
        if replay_buffer.ready() and total_steps % (train_agent_every_steps//num_envs) == 0 and total_steps*num_envs >= 0:
            if total_steps % (save_every_steps//num_envs) == 0:
                log_video = True
            else:
                log_video = False
            start_time = time.time()
            imagine_latent, agent_action, agent_logprob, agent_value, imagine_reward, imagine_termination = world_model_imagine_data(
                replay_buffer=replay_buffer,
                world_model=world_model,
                agent=agent,
                imagine_batch_size=imagine_batch_size,
                imagine_demonstration_batch_size=imagine_demonstration_batch_size,
                imagine_context_length=imagine_context_length,
                imagine_batch_length=imagine_batch_length,
                log_video=log_video,
                logger=logger
            )
            wandb.log({'runtime/imagine': time.time()-start_time})
            start_time = time.time()
            agent.update(
                latent=imagine_latent,
                action=agent_action,
                old_logprob=agent_logprob,
                old_value=agent_value,
                reward=imagine_reward,
                termination=imagine_termination,
                logger=logger
            )
            wandb.log({"runtime/agent": time.time()-start_time})
        # <<< train agent part

        # save model per episode
        if total_steps % (save_every_steps//num_envs) == 0:
            print(colorama.Fore.GREEN + f"Saving model at total steps {total_steps}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), f"ckpt/{name}/world_model_{total_steps}.pth")
            torch.save(agent.state_dict(), f"ckpt/{name}/agent_{total_steps}.pth")


def build_world_model(conf, action_dim, device: torch.device):
    return WorldModel(
        in_channels=conf.Models.WorldModel.InChannels,
        action_dim=action_dim,
        transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
        transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
        transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
        transformer_num_heads=conf.Models.WorldModel.TransformerNumHeads,
        device=device,
        conf=conf
    ).to(device)


def build_agent(conf, action_dim, device: torch.device):
    return agents.ActorCriticAgent(
        feat_dim=32*32+conf.Models.WorldModel.TransformerHiddenDim,
        num_layers=conf.Models.Agent.NumLayers,
        hidden_dim=conf.Models.Agent.HiddenDim,
        action_dim=action_dim,
        gamma=conf.Models.Agent.Gamma,
        lambd=conf.Models.Agent.Lambda,
        entropy_coef=conf.Models.Agent.EntropyCoef,
        device=device
    ).to(device)

def _wandb_init(cfg: DictConfig):
        ## Convert Omega Config to Wandb Config (letting wandb know of the config for current run)
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) ###TODO: check if model / global config ?
        expName = cfg.wandb.exp_name
        if cfg.wandb.log:
            mode = "online"
        else:
            mode = "disabled"

        wandb_run = wandb.init(config=config_dict, project=cfg.wandb.project_name, name=expName,
                                    mode=mode)  # wandb object has a set of configs associated with it as well 
        return wandb_run

@hydra.main(config_path="../GIT-STORM/config_files", config_name="STORM")
def main(conf: DictConfig):  
    # ignore warnings
    
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # set seed
    seed_np_torch(seed=conf.BasicSettings.Seed)
    # tensorboard writer
    logger = Logger(path=f"runs/{conf.BasicSettings.n}")
    device = torch.device(conf.BasicSettings.device)
    wandb_run = _wandb_init(conf)

    if conf.Task == "JointTrainAgent":
        # getting action_dim with dummy env
        dummy_env = build_single_env(conf.BasicSettings.env_name, conf.BasicSettings.ImageSize, seed=conf.BasicSettings.Seed)
        action_dim = dummy_env.action_space.n
        # build world model and agent
        world_model = build_world_model(conf, action_dim, device)
        world_model = torch.compile(world_model, mode="max-autotune")
        agent = build_agent(conf, action_dim, device)
        agent = torch.compile(agent, mode="max-autotune")
        # build replay buffer
        replay_buffer = ReplayBuffer(
            obs_shape=(conf.BasicSettings.ImageSize, conf.BasicSettings.ImageSize, 3),
            num_envs=conf.JointTrainAgent.NumEnvs,
            device=device,
            max_length=conf.JointTrainAgent.BufferMaxLength,
            warmup_length=conf.JointTrainAgent.BufferWarmUp,
            store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU,
        )
        # judge whether to load demonstration trajectory
        if conf.JointTrainAgent.UseDemonstration:
            print(colorama.Fore.MAGENTA + f"loading demonstration trajectory from {conf.BasicSettings.trajectory_path}" + colorama.Style.RESET_ALL)
            replay_buffer.load_trajectory(path=conf.BasicSettings.trajectory_path, device=conf.BasicSettings.device)
        # train
        joint_train_world_model_agent(
            env_name=conf.BasicSettings.env_name,
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_steps=conf.JointTrainAgent.SampleMaxSteps,
            image_size=conf.BasicSettings.ImageSize,
            replay_buffer=replay_buffer,
            world_model=world_model,
            agent=agent,
            train_dynamics_every_steps=conf.JointTrainAgent.TrainDynamicsEverySteps,
            train_agent_every_steps=conf.JointTrainAgent.TrainAgentEverySteps,
            batch_size=conf.JointTrainAgent.BatchSize,
            demonstration_batch_size=conf.JointTrainAgent.DemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0,
            batch_length=conf.JointTrainAgent.BatchLength,
            imagine_batch_size=conf.JointTrainAgent.ImagineBatchSize,
            imagine_demonstration_batch_size=conf.JointTrainAgent.ImagineDemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0,
            imagine_context_length=conf.JointTrainAgent.ImagineContextLength,
            imagine_batch_length=conf.JointTrainAgent.ImagineBatchLength,
            save_every_steps=conf.JointTrainAgent.SaveEverySteps,
            seed=conf.BasicSettings.Seed,
            logger=logger,
            name=conf.BasicSettings.n,
            device=device
        )
    else:
        raise NotImplementedError(f"Task {conf.Task} not implemented")

    # wandb_run.finish()

if __name__ == "__main__":
    main()