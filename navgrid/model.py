"""Example Multi-Layer Perceptron (MLP) policy showing how to wire a model into NavGrid."""

import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .config import Config


class NavGridMLPPolicy(nn.Module):
    architecture_version = "NavGrid-MLP-v1.0"
    """Example Multi-Layer Perceptron (MLP) policy demonstrating how to wire a custom model into NavGrid.

    Catches the two sequential inputs output by the environment:
      1. grid_input:  (B, T, H, W) -> flattened spatial grid view (11x11 = 121 cells)
      2. state_input: (B, T, STATE_DIM) -> continuous & discrete agent state vector (18 dims)
    Input dimension per timestep: (H * W + STATE_DIM) = (121 + 18) = 139 features.
    Across sequence length T (default T=64): flattened to (B, 139 * 64) = (B, 8896) features.
    """

    def __init__(
        self,
        grid_h: int = 11,
        grid_w: int = 11,
        state_dim: int = 18,
        seq_len: int = 64,
        hidden_dims: Tuple[int, ...] = (256, 128),
        num_actions: int = 4,
        num_directions: int = 4,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.step_dim = (grid_h * grid_w) + state_dim
        self.input_dim = self.step_dim * seq_len
        self.num_actions = num_actions
        self.num_directions = num_directions

        # Shared feature trunk
        layers = []
        in_dim = self.input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)

        # Policy & Value heads
        self.head_act = nn.Linear(in_dim, num_actions)
        self.head_dir = nn.Linear(in_dim, num_directions)
        self.head_val = nn.Linear(in_dim, 1)

    def forward(
        self,
        grid_input: torch.Tensor,
        state_input: torch.Tensor,
        pos_input: Optional[torch.Tensor] = None,
        collect_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if grid_input.ndim == 3:
            grid_input = grid_input.unsqueeze(1)
        if state_input.ndim == 2:
            state_input = state_input.unsqueeze(1)

        B, T, H, W = grid_input.shape
        grid_flat = grid_input.float().reshape(B, T, H * W)
        state_flat = state_input.float().reshape(B, T, -1)
        combined = torch.cat([grid_flat, state_flat], dim=-1)  # (B, T, step_dim)

        # Standardize temporal dimension to configured seq_len
        if T < self.seq_len:
            pad = torch.zeros(
                B, self.seq_len - T, self.step_dim,
                device=combined.device, dtype=combined.dtype
            )
            combined = torch.cat([pad, combined], dim=1)
        elif T > self.seq_len:
            combined = combined[:, -self.seq_len:]

        x = combined.reshape(B, self.input_dim)
        features = self.trunk(x)

        act_logits = self.head_act(features)
        dir_logits = self.head_dir(features)
        val_estimate = self.head_val(features).squeeze(-1)

        return act_logits, dir_logits, val_estimate


class PPOAgent(nn.Module):
    """PPO wrapper demonstrating how to wire a model into the NavGrid environment."""

    def __init__(self, config=None):
        super().__init__()
        self.cfg = config or Config
        self.random_action_policy = bool(getattr(self.cfg, "RANDOM_ACTION_POLICY", False))
        if self.random_action_policy:
            self.agent_model = None
            self.last_aux_dict = None
            return

        self.agent_model = NavGridMLPPolicy(
            grid_h=self.cfg.GRID_SIZE,
            grid_w=self.cfg.GRID_SIZE,
            state_dim=self.cfg.STATE_DIM,
            seq_len=self.cfg.SEQUENCE_LENGTH,
            hidden_dims=getattr(self.cfg, "MLP_HIDDEN_DIMS", (256, 128)),
            num_actions=self.cfg.NUM_ACTIONS,
            num_directions=self.cfg.NUM_DIRECTIONS,
        ).to(self.cfg.DEVICE)
        self.last_aux_dict = None

    def forward(self, grid_input, state_input, pos_input=None, collect_aux=False):
        B = state_input.shape[0]
        if self.random_action_policy:
            zeros = lambda n: torch.zeros(B, n, device=self.cfg.DEVICE)
            return (
                zeros(int(self.cfg.NUM_ACTIONS)),
                zeros(int(self.cfg.NUM_DIRECTIONS)),
                torch.zeros(B, device=self.cfg.DEVICE),
            )

        grid_input = grid_input.to(self.cfg.DEVICE)
        state_input = state_input.to(self.cfg.DEVICE)
        if pos_input is not None:
            pos_input = pos_input.to(self.cfg.DEVICE)

        act_logits, dir_logits, value = self.agent_model(
            grid_input, state_input, pos_input, collect_aux=collect_aux
        )
        if value.ndim == 2 and value.shape == (B, 1):
            value = value[:, 0]
        return act_logits, dir_logits, value

    def get_new_log_probs(
        self,
        state,
        action_idx,
        direction_idx,
        action_masks=None,
        direction_masks=None,
        collect_aux=False,
    ):
        if len(state) == 3:
            grid_inputs, state_inputs, pos_inputs = state
        else:
            grid_inputs, state_inputs = state
            pos_inputs = None

        a_logits, d_logits, values = self(
            grid_inputs, state_inputs, pos_inputs, collect_aux=collect_aux
        )

        if action_masks is not None:
            a_logits = a_logits.masked_fill(
                ~action_masks,
                torch.finfo(a_logits.dtype).min,
            )

        if direction_masks is not None:
            d_logits = d_logits.masked_fill(
                ~direction_masks,
                torch.finfo(d_logits.dtype).min,
            )

        a_dist = Categorical(logits=a_logits)
        d_dist = Categorical(logits=d_logits)

        a_logp = a_dist.log_prob(action_idx.to(self.cfg.DEVICE))
        d_logp = d_dist.log_prob(direction_idx.to(self.cfg.DEVICE))
        a_entropy = a_dist.entropy()
        d_entropy = d_dist.entropy()

        return a_logp, d_logp, values, a_entropy, d_entropy

    def get_value(self, state):
        if len(state) == 3:
            grid_inputs, state_inputs, pos_inputs = state
        else:
            grid_inputs, state_inputs = state
            pos_inputs = None
        _, _, values = self(grid_inputs, state_inputs, pos_inputs)
        return values

    def release_last_forward_graph(self) -> None:
        self.last_aux_dict = None

    def begin_router_state_observation_(self) -> None:
        pass

    def commit_router_state_observation_(self) -> None:
        pass

    def discard_router_state_observation_(self) -> None:
        pass

    def save(self, path=None):
        if self.random_action_policy:
            return
        if path is None:
            os.makedirs("models", exist_ok=True)
            path = self.cfg.MODEL_PATH
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.agent_model.state_dict(),
            "config": {
                "grid_h": self.cfg.GRID_SIZE,
                "grid_w": self.cfg.GRID_SIZE,
                "state_dim": self.cfg.STATE_DIM,
                "seq_len": self.cfg.SEQUENCE_LENGTH,
            },
        }, path)


    def calculate_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load(self, path=None):
        if path is None:
            path = self.cfg.MODEL_PATH
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return
        ckpt = torch.load(path, map_location=self.cfg.DEVICE)
        self.agent_model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded NavGrid MLP model weights from {path}")
