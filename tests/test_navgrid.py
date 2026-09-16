#!/usr/bin/env python3
"""Comprehensive unit and integration tests for NavGrid environment and example MLP model."""

import math
import os
import sys
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure headless mode during tests
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from navgrid import (
    Action,
    Agent,
    CellType,
    Config,
    Direction,
    GridEnvironment,
    NavGridMLPPolicy,
    ObstacleSetupChoice,
    PPO,
    PPOAgent,
    EnvironmentSimulation,
)


def test_wait_energy_and_terminal_penalty():
    """Verify that WAIT consumes 1.0 metabolic energy and applies incomplete-task penalty."""
    print("Testing WAIT energy and terminal penalty...")
    old_energy = Config.STARTING_ENERGY
    old_steps = Config.MAX_EPISODE_STEPS
    old_food = Config.SET_FOOD_COUNT
    old_scale = Config.SCALE_FOOD_COUNT
    old_obs = Config.OBSTACLE_CHOICE
    try:
        Config.STARTING_ENERGY = 1000.0
        Config.MAX_EPISODE_STEPS = 2
        Config.SET_FOOD_COUNT = 2
        Config.SCALE_FOOD_COUNT = False
        Config.OBSTACLE_CHOICE = ObstacleSetupChoice.EMPTY_NO_OBSTACLES

        env = GridEnvironment(0, size=11)
        agent = Agent(None, env, 0, 0)
        start_energy = agent.energy

        choice = {
            "action": torch.tensor(Action.WAIT.value),
            "direction": torch.tensor(0),
            "action_log_prob": 0.0,
            "direction_log_prob": 0.0,
            "direction_mask": agent.get_direction_mask(Action.WAIT.value),
        }

        first = agent.apply_action_choice(choice)
        second = agent.apply_action_choice(choice)

        assert agent.energy == start_energy - 2.0, f"Expected {start_energy - 2.0}, got {agent.energy}"
        assert not first[5], "First step should not terminate"
        assert second[5] and agent.time_limit_reached, "Second step should reach time limit"
        assert agent.reward_ledger["terminal"] == -10.0, f"Expected terminal penalty -10.0, got {agent.reward_ledger['terminal']}"
        print("  -> Passed: WAIT burns full energy and terminal penalty applies correctly.")
    finally:
        Config.STARTING_ENERGY = old_energy
        Config.MAX_EPISODE_STEPS = old_steps
        Config.SET_FOOD_COUNT = old_food
        Config.SCALE_FOOD_COUNT = old_scale
        Config.OBSTACLE_CHOICE = old_obs


def test_direction_entropy_normalizes_over_relevant_rows():
    """Verify that WAIT actions are excluded from direction entropy and loss normalization."""
    print("Testing WAIT-safe direction entropy normalization...")
    class DummyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_logits = nn.Parameter(torch.zeros(4))
            self.direction_logits = nn.Parameter(torch.tensor([1.0, 0.0, -1.0, 0.5]))
            self.last_aux_dict = None

        def get_new_log_probs(
            self, state, actions, directions, action_masks=None,
            direction_masks=None, collect_aux=False,
        ):
            batch = actions.shape[0]
            action_logits = self.action_logits.expand(batch, -1)
            direction_logits = self.direction_logits.expand(batch, -1)
            if action_masks is not None:
                action_logits = action_logits.masked_fill(
                    ~action_masks, torch.finfo(action_logits.dtype).min
                )
            if direction_masks is not None:
                direction_logits = direction_logits.masked_fill(
                    ~direction_masks, torch.finfo(direction_logits.dtype).min
                )
            action_dist = Categorical(logits=action_logits)
            direction_dist = Categorical(logits=direction_logits)
            return (
                action_dist.log_prob(actions),
                direction_dist.log_prob(directions),
                torch.zeros(batch),
                action_dist.entropy(),
                direction_dist.entropy(),
            )

    model = DummyPolicy()
    ppo = PPO(model, torch.optim.AdamW(model.parameters(), lr=1e-3))
    wait_mask = torch.tensor([True, False, False, False])
    move_mask = torch.ones(4, dtype=torch.bool)

    mb = {
        "state_inputs": (torch.zeros(2, 1), torch.zeros(2, 1)),
        "actions": torch.tensor([Action.WAIT.value, Action.MOVE.value]),
        "directions": torch.tensor([0, 0]),
        "action_masks": torch.ones(2, 4, dtype=torch.bool),
        "direction_masks": torch.stack((wait_mask, move_mask)),
        "old_action_log_probs": torch.full((2,), -math.log(4.0)),
        "old_direction_log_probs": torch.tensor([
            0.0,
            Categorical(logits=model.direction_logits.detach()).log_prob(torch.tensor(0)),
        ]),
        "old_values": torch.zeros(2),
        "returns": torch.zeros(2),
        "advantages": torch.ones(2),
    }

    _, _, _, info, total = ppo.compute_loss(mb)
    expected = Categorical(logits=model.direction_logits).entropy().item()
    assert abs(info["direction_entropy"] - expected) < 1e-6, f"Expected {expected}, got {info['direction_entropy']}"
    assert abs(info["direction_relevant_fraction"] - 0.5) < 1e-6, f"Expected 0.5 relevant fraction, got {info['direction_relevant_fraction']}"
    total.backward()
    assert torch.isfinite(model.direction_logits.grad).all()
    print("  -> Passed: WAIT rows correctly excluded from direction loss and entropy.")


def test_mlp_model_catches_input_sequences():
    """Verify that NavGridMLPPolicy catches (B, T, 11, 11) and (B, T, 18) sequences."""
    print("Testing NavGridMLPPolicy and PPOAgent input catching...")
    B, T = 4, 64
    agent = PPOAgent()
    grid_views = torch.randint(1, 13, (B, T, 11, 11))
    state_vectors = torch.randn(B, T, Config.STATE_DIM)
    pos_coords = torch.randint(0, 11, (B, T, 2))

    act_logits, dir_logits, value = agent(grid_views, state_vectors, pos_coords)
    assert act_logits.shape == (B, 4), f"Unexpected act_logits shape: {act_logits.shape}"
    assert dir_logits.shape == (B, 4), f"Unexpected dir_logits shape: {dir_logits.shape}"
    assert value.shape == (B,), f"Unexpected value shape: {value.shape}"

    # Verify log_prob computation
    actions = torch.randint(0, 4, (B,))
    directions = torch.randint(0, 4, (B,))
    a_logp, d_logp, vals, a_ent, d_ent = agent.get_new_log_probs(
        (grid_views, state_vectors, pos_coords), actions, directions
    )
    assert torch.isfinite(a_logp).all()
    assert torch.isfinite(d_logp).all()
    assert torch.isfinite(vals).all()

    loss = a_logp.sum() + d_logp.sum() + vals.sum()
    loss.backward()
    for name, param in agent.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
    print("  -> Passed: NavGridMLPPolicy catches sequential inputs and propagates finite gradients.")


def test_environment_and_agent_interaction():
    """Verify GridEnvironment creation, item distribution, and Agent step mechanics."""
    print("Testing GridEnvironment and Agent step mechanics...")
    env = GridEnvironment(0, size=11)
    agent = Agent(None, env, 0, 0)
    assert env.is_within_bounds(agent.x, agent.y)

    mask = agent.get_action_mask()
    assert mask[Action.MOVE.value].item() is True
    assert mask[Action.WAIT.value].item() is True

    # Test take action
    action_logits = torch.randn(4)
    direction_logits = torch.randn(4)
    choice = agent.sample_action_choice(action_logits, direction_logits)
    assert "action" in choice and "direction" in choice
    print("  -> Passed: GridEnvironment and Agent step mechanics operating normally.")



def test_end_to_end_simulation():
    """Verify EnvironmentSimulation runs rollouts and completes PPO update cleanly."""
    print("Testing EnvironmentSimulation end-to-end rollout and update...")
    import threading
    Config.HEADLESS = True
    Config.VISUALIZE = False
    Config.NUM_ENVS = 1
    Config.ENVIRONMENT_SIZE = 11
    Config.PPO_EPISODES_PER_UPDATE = 1
    Config.MINIBATCH_SIZE = 8
    Config.MAX_EPISODE_STEPS = 16
    Config.CURRICULUM_ENABLED = False

    sim = EnvironmentSimulation()
    stop_event = threading.Event()
    sim.run_episode(stop_event, None, 0)

    assert sim.update_count >= 1, f"Expected at least 1 PPO update, got {sim.update_count}"
    assert sim.last_rollout_batch_samples > 0, "Expected rollout batch samples > 0"
    print(f"  -> Successfully ran episode and executed {sim.update_count} PPO updates.")
    print(f"  -> Last rollout batch samples: {sim.last_rollout_batch_samples}, last losses: {sim.last_losses}")
    print("  -> Passed: End-to-end simulation and PPO update operate cleanly.")


def test_egocentric_vs_global_view_switch():
    """Verify GridEnvironment get_view supports both egocentric and global view modes."""
    print("Testing Egocentric vs Global view modes...")
    env = GridEnvironment(0, size=11)
    agent = Agent(None, env, 0, 0)

    # 1. Egocentric (default: USE_GLOBAL_STATE = False)
    Config.USE_GLOBAL_STATE = False
    ego_view = env.get_view(center_r=agent.x, center_c=agent.y, agent_r=agent.x, agent_c=agent.y, view_size=11)
    assert ego_view.shape == (11, 11), f"Expected (11, 11), got {ego_view.shape}"
    # In egocentric view, center cell (5, 5) represents the requesting agent
    half = 11 // 2
    assert ego_view[half, half] == CellType.AGENT.value, f"Expected center to be AGENT (11), got {ego_view[half, half]}"

    # 2. Global view (switch: USE_GLOBAL_STATE = True)
    Config.USE_GLOBAL_STATE = True
    global_view = env.get_view(center_r=0, center_c=0, agent_r=agent.x, agent_c=agent.y, view_size=11)
    assert global_view.shape == (11, 11), f"Expected (11, 11), got {global_view.shape}"
    assert global_view[agent.x, agent.y] == CellType.AGENT.value, f"Expected agent at ({agent.x}, {agent.y}), got {global_view[agent.x, agent.y]}"

    # Reset back to default
    Config.USE_GLOBAL_STATE = False
    print("  -> Passed: Both egocentric and global view modes operate correctly behind the switch.")


def test_multi_environment_rollout():
    """Verify EnvironmentSimulation runs rollouts across multiple parallel environments."""
    print("Testing Multi-Environment Rollout (NUM_ENVS = 2)...")
    Config.DEVICE = "cpu"
    Config.HEADLESS = True
    Config.VISUALIZE = False
    Config.NUM_ENVS = 2
    Config.NUM_AGENTS = 1
    Config.ENVIRONMENT_SIZE = 11
    Config.PPO_EPISODES_PER_UPDATE = 1
    Config.MINIBATCH_SIZE = 8
    Config.MAX_EPISODE_STEPS = 16
    Config.CURRICULUM_ENABLED = False

    sim = EnvironmentSimulation()
    assert len(sim.envs) == 2, f"Expected 2 environments, got {len(sim.envs)}"
    assert len(sim.agents) == 2, f"Expected 2 agents (1 per env), got {len(sim.agents)}"

    stop_event = threading.Event()
    sim.run_episode(stop_event, None, 0)

    assert sim.update_count >= 1, f"Expected at least 1 PPO update, got {sim.update_count}"
    assert sim.last_rollout_batch_samples > 0, "Expected rollout batch samples > 0"
    print(f"  -> Successfully ran parallel multi-env episode and executed {sim.update_count} PPO updates.")
    print("  -> Passed: Multi-environment rollout executes and trains cleanly.")

    # Reset back to default
    Config.NUM_ENVS = 1


if __name__ == "__main__":
    print("=" * 60)
    print("Running NavGrid Test Suite")
    print("=" * 60)
    test_wait_energy_and_terminal_penalty()
    test_direction_entropy_normalizes_over_relevant_rows()
    test_mlp_model_catches_input_sequences()
    test_environment_and_agent_interaction()
    test_end_to_end_simulation()
    test_egocentric_vs_global_view_switch()
    test_multi_environment_rollout()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
