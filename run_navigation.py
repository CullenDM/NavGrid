#!/usr/bin/env python3
"""Main training and evaluation entrypoint for NavGrid."""

from __future__ import annotations

import argparse
import gc
import os
import sys
import torch

from navgrid import Config, EnvironmentSimulation


def parse_args():
    parser = argparse.ArgumentParser(description="NavGrid Training & Simulation Runner")
    parser.add_argument("--headless", action="store_true", help="Run without Pygame rendering")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda)")
    parser.add_argument("--episodes", type=int, default=None, help="Episodes per PPO update")
    parser.add_argument("--minibatch", type=int, default=None, help="Minibatch size for PPO")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--fps", type=int, default=None, help="Target FPS for visualizer")
    parser.add_argument("--random-policy", action="store_true", help="Run with uniform random legal actions (no model)")
    parser.add_argument("--global-view", action="store_true", help="Use full-grid global view instead of egocentric observation window")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of parallel environments for rollouts")
    parser.add_argument("--num-agents", type=int, default=None, help="Number of agents per environment")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.headless:
        Config.HEADLESS = True
        Config.VISUALIZE = False
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    if args.device:
        Config.DEVICE = args.device

    if args.episodes is not None:
        Config.PPO_EPISODES_PER_UPDATE = args.episodes

    if args.minibatch is not None:
        Config.MINIBATCH_SIZE = args.minibatch

    if args.lr is not None:
        Config.LEARNING_RATE = args.lr

    if args.fps is not None:
        Config.FPS = args.fps

    if args.random_policy:
        Config.RANDOM_ACTION_POLICY = True
    if args.global_view:
        Config.USE_GLOBAL_STATE = True
    if args.num_envs is not None:
        Config.NUM_ENVS = args.num_envs
    if args.num_agents is not None:
        Config.NUM_AGENTS = args.num_agents

    print("=" * 60)
    print(" NavGrid: Standalone RL Navigation Environment")
    print("=" * 60)
    print(f" Device:         {Config.DEVICE}")
    print(f" Environments:   {Config.NUM_ENVS} parallel env(s), {Config.NUM_AGENTS} agent(s)/env")
    print(f" Environment:    {Config.ENVIRONMENT_SIZE}x{Config.ENVIRONMENT_SIZE} (View: {Config.GRID_SIZE}x{Config.GRID_SIZE}, Mode: {'Global' if Config.USE_GLOBAL_STATE else 'Egocentric'})")
    print(f" State Dim:      {Config.STATE_DIM} (Sequence: {Config.SEQUENCE_LENGTH})")
    print(f" Model:          {Config.MODEL_NAME}")
    print(f" Visualizer:     {'Disabled (Headless)' if Config.HEADLESS else f'Active ({Config.FPS} FPS)'}")
    print("=" * 60, flush=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    simulation = EnvironmentSimulation()
    try:
        simulation.run()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user. Exiting cleanly.")


if __name__ == "__main__":
    main()
