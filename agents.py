import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy

from sub_models.functions_losses import SymLogTwoHotLoss
from utils import EMAScalar
import wandb


def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage * len(flat_x))
    per = torch.kthvalue(flat_x, kth).values
    return per


def calc_lambda_return(
    rewards, values, termination, gamma, lam, device: torch.device, dtype=torch.float32
):
    # Invert termination to have 0 if the episode ended and 1 otherwise
    inv_termination = (termination * -1) + 1

    batch_size, batch_length = rewards.shape[:2]
    gamma_return = torch.zeros(
        (batch_size, batch_length + 1), dtype=dtype, device=device
    )
    gamma_return[:, -1] = values[:, -1]
    for t in reversed(range(batch_length)):  # with last bootstrap
        gamma_return[:, t] = (
            rewards[:, t]
            + gamma * inv_termination[:, t] * (1 - lam) * values[:, t]
            + gamma * inv_termination[:, t] * lam * gamma_return[:, t + 1]
        )
    return gamma_return[:, :-1]


class ActorCriticAgent(nn.Module):
    def __init__(
        self,
        feat_dim,
        num_layers,
        hidden_dim,
        action_dim,
        gamma,
        lambd,
        entropy_coef,
        device: torch.device,
        dist: "str" = "categorical",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.lambd = lambd
        self.entropy_coef = entropy_coef
        self.use_amp = True
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.device = device
        self.dist = dist

        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)
        self._min_std = 0.1
        self._max_std = 1
        actor = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ]
        for i in range(num_layers - 1):
            actor.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                ]
            )
        if self.dist == "categorical":
            self.actor = nn.Sequential(*actor, nn.Linear(hidden_dim, action_dim)) 
        elif self.dist == "normal":
            self.actor = nn.Sequential(*actor, nn.Linear(hidden_dim, action_dim*2))
        elif self.dist == "tanh":
            self.actor = nn.Sequential(*actor, nn.Linear(hidden_dim, action_dim), nn.Tanh())
        elif self.dist == "trunc_normal":
            self.actor = nn.Sequential(*actor, nn.Linear(hidden_dim, action_dim*2))

        critic = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ]
        for i in range(num_layers - 1):
            critic.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                ]
            )

        self.critic = nn.Sequential(*critic, nn.Linear(hidden_dim, 255))
        self.slow_critic = copy.deepcopy(self.critic)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, eps=1e-5)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        for slow_param, param in zip(
            self.slow_critic.parameters(), self.critic.parameters()
        ):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))

    def policy(self, x):
        logits = self.actor(x)
        return logits

    def value(self, x):
        value = self.critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    @torch.no_grad()
    def slow_value(self, x):
        value = self.slow_critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    def get_logits_raw_value(self, x):
        logits = self.actor(x)
        raw_value = self.critic(x)
        return logits, raw_value

    @torch.no_grad()
    def sample(self, latent, greedy=False):
        self.eval()
        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp
        ):
            logits = self.policy(latent)
            if self.dist == "categorical":
                dist = distributions.Categorical(logits=logits)
                if greedy:
                    action = dist.probs.argmax(dim=-1)
                else:
                    action = dist.sample()
            elif self.dist == "normal":
                mu, sigma = torch.chunk(logits, 2, dim=-1)
                dist = distributions.Normal(loc=mu, scale=sigma)
                if greedy:
                    action = mu
                else:
                    action = dist.sample()
            elif self.dist == "tanh":
                action = logits
            elif self.dist == "trunc_normal":
                mean, std = torch.chunk(logits, 2, dim=-1)
                mean = torch.tanh(mean)
                std = 2 * torch.sigmoid(std / 2) + self._min_std
                dist = SafeTruncatedNormal(mean, std, -1, 1)
                # print(f"dist from safetrunc{dist.sample()}")
                dist = ContDist(distributions.independent.Independent(dist, 1))
                if greedy:
                    action = mean
                else:
                    action = dist.sample()
                    
        return action

    def sample_as_env_action(self, latent, greedy=False):
        action = self.sample(latent, greedy)
        return action.detach().squeeze(0).cpu().numpy() if self.dist == "categorical" else action.detach().float().squeeze(0).cpu().numpy()

    def update(
        self, latent, action, old_logprob, old_value, reward, termination, logger=None
    ):
        """
        Update policy and value model
        """
        self.train()
        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp
        ):
            logits, raw_value = self.get_logits_raw_value(latent)
            if self.dist == "categorical":
                dist = distributions.Categorical(logits=logits[:, :-1])
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
            elif self.dist == "tanh":
                entropy = 0
            elif self.dist == "trunc_normal":
                mean, std = torch.chunk(logits[:, :-1], 2, dim=-1)
                mean = torch.tanh(mean)
                std = 2 * torch.sigmoid(std / 2) + self._min_std
                dist = SafeTruncatedNormal(mean, std, -1, 1)
                dist = ContDist(distributions.independent.Independent(dist, 1))
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

            # decode value, calc lambda return
            slow_value = self.slow_value(latent)
            slow_lambda_return = calc_lambda_return(
                reward, slow_value, termination, self.gamma, self.lambd, self.device,
            )
            value = self.symlog_twohot_loss.decode(raw_value)
            lambda_return = calc_lambda_return(
                reward, value, termination, self.gamma, self.lambd, self.device,
            )

            # update value function with slow critic regularization
            value_loss = self.symlog_twohot_loss(
                raw_value[:, :-1], lambda_return.detach()
            )
            slow_value_regularization_loss = self.symlog_twohot_loss(
                raw_value[:, :-1], slow_lambda_return.detach()
            )
            lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
            upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
            S = upper_bound - lower_bound
            norm_ratio = torch.max(
                torch.ones(1).to(self.device), S
            )  # max(1, S) in the paper
            norm_advantage = (lambda_return - value[:, :-1]) / norm_ratio
            policy_loss = -(log_prob * norm_advantage.detach()).mean()
            # policy_loss = -(norm_advantage).mean()
            # policy_loss = -(S).mean()

            entropy_loss = entropy.mean()

            loss = (
                policy_loss
                + value_loss
                + slow_value_regularization_loss
                - self.entropy_coef * entropy_loss
            )

        # gradient descent
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.update_slow_critic()

        if logger is not None:
            logger.log("ActorCritic/policy_loss", policy_loss.item())
            logger.log("ActorCritic/value_loss", value_loss.item())
            logger.log("ActorCritic/entropy_loss", entropy_loss.item())
            logger.log("ActorCritic/S", S.item())
            logger.log("ActorCritic/norm_ratio", norm_ratio.item())
            logger.log("ActorCritic/total_loss", loss.item())
        
        loss_dict = {
            "ActorCritic/policy_loss": policy_loss.item(),
            "ActorCritic/value_loss": value_loss.item(),
            "ActorCritic/entropy_loss": entropy_loss.item(),
            "ActorCritic/S": S.item(),
            "ActorCritic/norm_ratio": norm_ratio.item(),
            "ActorCritic/total_loss": loss.item()
        }

        wandb.log(loss_dict)

class SafeTruncatedNormal(distributions.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class ContDist:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        return self._dist.mean

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        return self._dist.log_prob(x)