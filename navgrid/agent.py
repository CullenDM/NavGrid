"""Agent state, inventory, action execution, and transition storage for NavGrid."""

from __future__ import annotations

import copy
import math
import os
import random
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch.distributions import Categorical

from .config import Config
from .constants import (
    CARDINAL_DELTAS,
    Action,
    CellType,
    Direction,
    EpisodeVariant,
)
from .environment import GridEnvironment, uses_competitive_food_share_completion

if TYPE_CHECKING:
    from .visualizer import GridVisualizer

class Agent:
    """Agent that navigates the grid, manages inventory, and collects PPO transitions."""
    def _symlog(self, v):
        return math.copysign(math.log1p(abs(v)), v)

    def _sat(self, v, tau):
        return v / (v + tau)

    
    # -------------------------------
    # Reward shaping (maximal minus modifications)
    # -------------------------------
    FOOD_REWARD = 1.0
    COMPLETION_REWARD = 100.0
    
    # Movement shaping (empty/unvisited)
    # Retracing is valid locomotion. Novel and revisited cells carry the same
    # movement cost; exploration pressure comes from food-directed potential.
    UNVISITED_EMPTY_REWARD = -0.002
    EMPTY_REWARD = -0.002
    FOOD_DISTANCE_POTENTIAL_SCALE = 0.005
    
    # Optional extras (now active)
    ADJACENT_FOOD_NOT_EATEN_REWARD = 0 # -0.0010  # walked away while food was adjacent
    STEPS_SINCE_FOOD_REWARD = 0 # -0.000008        # linear impatience until next bite
    DIRECTION_MAINTAINED_REWARD =  0# 0.020      # added as (value / direction_counter)
    STEP_REWARD = -0.001                    # tiny time pressure
    REDUNDANT_MOVEMENT_REWARD = 0#-0.010       # didn’t change cells
    NO_FOOD_LEFT_REWARD = 0.0                # episode ends anyway
    
    # Collisions / others
    BOUNCING_COLLISION_REWARD = 0#-0.020
    OTHER_AGENT_REWARD = 0#-0.020              # stepping into another agent (if it ever happens)
    OBSTACLE_REWARD = 0#-0.020                    # stepping into static is blocked elsewhere
    LOOPING_OBSTACLE_REWARD = 0.0
    COUNTER_LOOPING_OBSTACLE_REWARD = 0.0
    WANDERING_OBSTACLE_REWARD = 0.0
    
    # -------------------------------
    # Fine-grained invalid move penalties
    # -------------------------------
    # Slightly larger negatives for fundamentally unmovable/forbidden targets
    INVALID_STATIC_OBSTACLE_PENALTY = -0.020
    INVALID_GRABBABLE_OBSTACLE_PENALTY = -0.020
    INVALID_OTHER_AGENT_PENALTY = -0.020
    # Dynamic obstacles are milder
    INVALID_DYNAMIC_OBSTACLE_PENALTY = -0.020
    # Out-of-bounds (boundary hit)
    BOUNDARY_REWARD = -0.030
    
    # -------------------------------
    # Moveable obstacle interaction
    # -------------------------------
    # Successful push: decays with count via  / num_moveable_obstacle_interactions
    MOVEABLE_OBSTACLE_REWARD = 0#0.020
    # Failed/blocked push: teeny negative
    MOVEABLE_OBSTACLE_FAIL_REWARD = -0.010
    
    # -------------------------------
    # Grabbable/placeable interactions
    # -------------------------------
    GRABBABLE_OBSTACLE_REWARD = 0#0.010
    PLACEABLE_OBSTACLE_REWARD = 0#0.010
    GRABBABLE_OBSTACLE_FAIL_REWARD = -0.010
    PLACEABLE_OBSTACLE_FAIL_REWARD = -0.010
    
    # Catch-all for any remaining invalid movement cases
    MOVEMENT_FAILURE_REWARD =  -0.020
    
    # Episode penalties/bonuses
    # Energy death scales this maximum by the fraction of food left. Progress
    # reduces the terminal cost continuously instead of erasing it after one bite.
    ENERGY_DEATH_MAX_PENALTY = -10.0
    # A step-cap termination with food remaining is the same incomplete task as
    # starving. Without this, WAIT can buy a nearly free, non-terminal timeout.
    TIME_LIMIT_MAX_PENALTY = -10.0

    # -------------------------------
    # Arena predation (agent versus agent)
    # -------------------------------
    # Moving onto a living opponent kills it. The killer's reward matches a
    # single bite and never chains: FOOD_REWARD flat, with no context-food
    # multiplier applied.
    AGENT_KILL_REWARD = FOOD_REWARD
    # Pinned to the snail's terminal rather than aliasing the starvation
    # penalty. Being killed by a rival is the same kind of event as being caught
    # by the pursuer; starving with nothing eaten is a different one, and it has
    # since been softened independently.
    AGENT_KILL_DEATH_REWARD = -10
    # A kill is worth a tenth of a meal. Held as a fraction of the agent's own
    # energy_for_food so it tracks the food energy automatically and never has
    # to be set by hand when the grid or the energy model changes.
    AGENT_KILL_ENERGY_FRACTION = 0.10
    
    # -------------------------------
    # Fixed energy model. World size changes task geometry, not metabolism.
    # -------------------------------
    ENERGY_CONSUMPTION_RATE = 1
    # Waiting advances the world and costs one full metabolic tick. Its utility
    # must come from timing a dynamic event, not from extending survival.
    ENERGY_WAIT_FRACTION = 1.0
    
    # Extra energy costs for specific interactions
    ENERGY_EXTRA_FOR_PUSH_SUCCESS = .5
    ENERGY_EXTRA_FOR_PUSH_FAIL = 1

    # NEW: make GRAB/PLACE extras configurable (multipliers of ENERGY_CONSUMPTION_RATE)
    ENERGY_EXTRA_FOR_GRAB_SUCCESS = 0.5
    ENERGY_EXTRA_FOR_GRAB_FAIL = 1.0
    ENERGY_EXTRA_FOR_PLACE_SUCCESS = 0.5
    ENERGY_EXTRA_FOR_PLACE_FAIL = 1.0

    ENERGY_PENALTY_BOUNCING_COLLISION = 10

    VIEW_SEQUENCE_LENGTH = Config.SEQUENCE_LENGTH

    def __init__(self, model, environment, env_idx, agent_idx, visualizer=None):
        self.model = model
        self.env = environment
        self.env_idx = env_idx
        self.agent_idx = agent_idx
        self.visualizer = visualizer
        
        self.max_energy = float(Config.STARTING_ENERGY)
        self.energy_for_food = float(Config.FOOD_ENERGY)
        self.energy = self.max_energy

        self.start_x, self.start_y = None, None
        self.transitions = TransitionStorage()
        self.reset()

    # -------------------------------
    # Lifecycle
    # -------------------------------
    def reset(self, participating: bool = True):
        self.max_energy = float(Config.STARTING_ENERGY)
        self.energy_for_food = float(Config.FOOD_ENERGY)
        if self.max_energy <= 0.0 or self.energy_for_food < 0.0:
            raise ValueError("STARTING_ENERGY must be positive and FOOD_ENERGY nonnegative")
        self.participating = bool(participating)
        if not self.participating:
            # Sits this episode out. Never registered, so it never appears on
            # the board even for a frame, is not hunted, and cannot be collided
            # with. Everything below still runs to clear last episode's state.
            pass
        elif self.env.new_grid_generated or self.start_x is None:
            spawnable = self.env.get_spawnable_cells(allow_fallback=False)
            if not spawnable:
                raise RuntimeError(
                    f"Env {self.env.idx} has no valid strict spawn cell"
                )
            self.x, self.y = random.choice(tuple(spawnable))
            self.start_x, self.start_y = self.x, self.y
        else:
            self.x, self.y = self.start_x, self.start_y

        if self.participating:
            self.env.register_agent(self, (self.x, self.y))
        self.energy = self.max_energy
        self.x_prev, self.y_prev = self.x, self.y
        self.pre_action_position = (self.x, self.y)

        self.num_grabbable_obstacle_interactions = 0
        self.num_moveable_obstacle_interactions = 0  # properly used now
        self.num_placeable_obstacle_interactions = 0

        self.steps_since_last_food = 0
        self.stalled_ticks = 0
        self.total_reward = 0
        self.food_eaten = 0
        # Per-episode choice histograms, for watching WAIT adoption over training.
        self.action_counts = [0] * Config.NUM_ACTIONS
        self.direction_counts = [0] * Config.NUM_DIRECTIONS
        self.food_reward_multipliers = []
        self.done = not self.participating
        self.died = False
        self.killed_by_agent = False
        self.kills = 0
        self.pending_agent_kill = 0
        self.reward_ledger = {
            "predation": 0.0,
            "step": 0.0,
            "movement": 0.0,
            "invalid_move": 0.0,
            "interaction_success": 0.0,
            "interaction_fail": 0.0,
            "food": 0.0,
            "completion": 0.0,
            "terminal": 0.0,
            "blocked_streak": 0.0,
            "distance_shaping": 0.0,
        }
        self.direction = None
        self.last_direction = None
        self.direction_counter = 0
        self.last_action = None
        self.previous_action = None
        self.inventory = []
        self.steps = 0
        self.state_vector = None
        self.last_view = None
        self.context_length = int(
            getattr(self.env, "sequence_length", Config.SEQUENCE_LENGTH)
        )
        if not 1 <= self.context_length <= int(self.VIEW_SEQUENCE_LENGTH):
            raise ValueError(
                f"Environment {self.env_idx} context length {self.context_length} "
                f"must be in [1, {self.VIEW_SEQUENCE_LENGTH}]"
            )
        self.init_sequences()
        self.total_predicted_rewards = 0
        self.last_reward = 0
        self.last_completion_reward = 0.0
        self.pending_food_potential = None

        # stats collected by PPO runner
        self.value_loss = []
        self.action_loss = []
        self.direction_loss = []
        self.mean_action_loss = 0
        self.mean_direction_loss = 0
        self.mean_value_loss = 0
        self.compute_times = []
        self.mean_compute_time = 0
        self.approx_kls = []
        self.clip_fracs = []
        self.policy_entropies = []

        self.visited_cells = set()
        if self.participating:
            self.visited_cells.add((self.x, self.y))
        self.collided_with_bouncing_obstacle = False
        self.pending_bouncer_hits = 0
        self.last_bouncer_id = None
        self.pending_snail_hits = 0
        self.killed_by_snail = False
        self.ate_last_step = False
        self.blocked_last_step = False
        self.death_penalty_applied = False
        self.time_limit_reached = False

        rate = self.ENERGY_CONSUMPTION_RATE
        self._tau_hunger = (self.max_energy / rate) if rate > 0 else 1000.0
        if Config.MAX_EPISODE_STEPS > 0:
            self._t_max = float(Config.MAX_EPISODE_STEPS)
        else:
            self._t_max = ((self.max_energy + self.env.num_food * self.energy_for_food) / rate) if rate > 0 else 1000.0
        food_reward_ceiling = (
            self.FOOD_REWARD
            * max(1, self.env.initial_food_count)
            * (
                float(Config.FOOD_CHAIN_MAX_MULTIPLIER)
                if Config.USE_CONTEXT_FOOD_CHAIN_BONUS
                else 1.0
            )
        )
        r_ref = self.completion_reward() + food_reward_ceiling
        self._r_norm_denom = self._symlog(1.0 + r_ref)

    # -------------------------------
    # Sequences / state
    # -------------------------------
    def get_state_vector(self):
        last_act_oh = [0.0] * Config.NUM_ACTIONS
        if self.steps > 0 and self.last_action is not None:
            last_act_oh[int(self.last_action)] = 1.0

        last_dir_oh = [0.0] * Config.NUM_DIRECTIONS
        if self.direction is not None:
            last_dir_oh[int(self.direction)] = 1.0

        state = [
            self.energy / (self.energy + self.max_energy),
            self._sat(self.steps_since_last_food, self._tau_hunger),
            len(self.inventory) / Config.MAX_INVENTORY,
            self.food_eaten / (self.env.initial_food_count + 1e-6),
            self._sat(self.steps, self._t_max),
            max(-1.0, min(1.0, self._symlog(self.last_reward) / self._r_norm_denom)),
            max(-1.0, min(1.0, self._symlog(self.total_reward / (self.steps + 1)) / self._r_norm_denom)),
            float(self.collided_with_bouncing_obstacle),
            float(self.ate_last_step),
            float(self.blocked_last_step),
            *last_act_oh,
            *last_dir_oh
        ]
        state_vector = torch.tensor(state, dtype=torch.float32, device=Config.DEVICE)
        self.state_vector = state_vector
        self.state_vector_sequence.append(state_vector)
        self.collided_with_bouncing_obstacle = False
        self.ate_last_step = False
        self.blocked_last_step = False
        return state_vector

    def init_sequences(self):
        self.view_sequence = deque(
            [torch.full((Config.GRID_SIZE, Config.GRID_SIZE), CellType.BOUNDARY.value, dtype=torch.float32, device=Config.DEVICE)
             for _ in range(Config.SEQUENCE_LENGTH)],
            maxlen=self.VIEW_SEQUENCE_LENGTH,
        )
        self.state_vector_sequence = deque(
            [torch.zeros(Config.STATE_DIM, dtype=torch.float32, device=Config.DEVICE) for _ in range(Config.SEQUENCE_LENGTH)],
            maxlen=self.VIEW_SEQUENCE_LENGTH,
        )
        self.pos_sequence = deque(
            [torch.tensor([self.x, self.y], dtype=torch.long, device=Config.DEVICE) for _ in range(Config.SEQUENCE_LENGTH)],
            maxlen=self.VIEW_SEQUENCE_LENGTH,
        )
        self.state_vector = self.state_vector_sequence[-1]

    def _mask_to_context(
        self,
        sequence: torch.Tensor,
        padding: torch.Tensor,
    ) -> torch.Tensor:
        """Left-pad history so every PPO sample retains the configured max shape."""
        hidden = int(sequence.shape[0]) - int(self.context_length)
        if hidden <= 0:
            return sequence
        masked = sequence.clone()
        masked[:hidden] = padding.to(device=masked.device, dtype=masked.dtype)
        return masked

    def prepare_view_tensor(self, update_view=True):
        if update_view:
            center_x, center_y = (self.env.size // 2, self.env.size // 2) if Config.USE_GLOBAL_STATE else (self.x, self.y)
            current_view = self.env.get_view(center_x, center_y, self.x, self.y, Config.GRID_SIZE)
            current_view_tensor = torch.from_numpy(current_view).float().to(Config.DEVICE)
            self.view_sequence.append(current_view_tensor)
            self.pos_sequence.append(torch.tensor([self.x, self.y], dtype=torch.long, device=Config.DEVICE))
            self.last_view = current_view
        view_tensor_sequence = torch.stack(list(self.view_sequence), dim=0)
        padding = torch.full(
            (Config.GRID_SIZE, Config.GRID_SIZE),
            CellType.BOUNDARY.value,
            device=view_tensor_sequence.device,
            dtype=view_tensor_sequence.dtype,
        )
        return self._mask_to_context(view_tensor_sequence, padding)

    def prepare_pos_sequence(self):
        pos_sequence = torch.stack(list(self.pos_sequence), dim=0)
        return self._mask_to_context(
            pos_sequence,
            torch.zeros(2, device=pos_sequence.device, dtype=pos_sequence.dtype),
        )

    def prepare_state_vector_sequence(self, update_state=True):
        if update_state:
            self.get_state_vector()
        state_vector_sequence = torch.stack(list(self.state_vector_sequence), dim=0)
        return self._mask_to_context(
            state_vector_sequence,
            torch.zeros(
                Config.STATE_DIM,
                device=state_vector_sequence.device,
                dtype=state_vector_sequence.dtype,
            ),
        )

    
    def get_direction_mask(self, action: int) -> torch.Tensor:
        mask = torch.zeros(
            Config.NUM_DIRECTIONS,
            dtype=torch.bool,
            device=Config.DEVICE,
        )

        if action == Action.MOVE.value:
            # Intentionally retain all four directions.
            # Boundary hits, blocked moves and failed pushes remain learnable.
            mask[:] = True
            return mask

        if action == Action.GRAB.value:
            if len(self.inventory) >= Config.MAX_INVENTORY:
                return mask

            adjacent = set(
                self.env.get_adjacent_grabbable_obstacles(self.x, self.y)
            )
            for d, (dr, dc) in enumerate(CARDINAL_DELTAS):
                mask[d] = (self.x + dr, self.y + dc) in adjacent
            return mask

        if action == Action.WAIT.value:
            # Direction is meaningless for WAIT. Pin it to one legal index so the
            # sampled log-prob is deterministic (log 1 = 0) instead of injecting
            # noise into the direction head on every held turn.
            mask[0] = True
            return mask

        if action == Action.PLACE.value:
            if not self.inventory:
                return mask

            for d, (dr, dc) in enumerate(CARDINAL_DELTAS):
                r, c = self.x + dr, self.y + dc
                mask[d] = bool(
                    self.env.is_within_bounds(r, c)
                    and self.env.grid[r, c] == CellType.EMPTY
                )
            return mask

        return mask

    def get_action_mask(self) -> torch.Tensor:
        # Built from the live action space, never a literal list: a hardcoded
        # width here silently disagrees with a 3-action checkpoint and only
        # surfaces as a masked_fill size error at sampling time.
        availability = {
            Action.MOVE.value: True,   # MOVE is always unmasked in canonical V25e.
            Action.GRAB.value: any(self.get_direction_mask(Action.GRAB.value).tolist()),
            Action.PLACE.value: any(self.get_direction_mask(Action.PLACE.value).tolist()),
            Action.WAIT.value: True,   # holding position is always available
        }
        mask = torch.tensor(
            [bool(availability.get(i, False)) for i in range(Config.NUM_ACTIONS)],
            dtype=torch.bool, device=Config.DEVICE,
        )
        assert mask.any()
        return mask

    def update(self, action_logits, direction_logits, view_tensor_sequence, state_vector_sequence, pos_sequence, value):
        action_mask = self.get_action_mask()
        action, direction, action_log_prob, direction_log_prob, reward, done, direction_mask = self.take_action(
            action_logits, direction_logits, action_mask=action_mask
        )
        return (view_tensor_sequence, state_vector_sequence, pos_sequence, action_mask, direction_mask), action, direction, action_log_prob, direction_log_prob, value, reward, done

    def sample_action_choice(
        self,
        action_logits,
        direction_logits,
        *,
        action_mask: torch.Tensor = None,
        deterministic: bool = False,
    ) -> dict:
        """Choose from one immutable pre-action state without mutating the arena."""
        if action_mask is None:
            action_mask = self.get_action_mask()

        masked_action_logits = action_logits.masked_fill(
            ~action_mask,
            torch.finfo(action_logits.dtype).min,
        )

        action_dist = Categorical(logits=masked_action_logits)
        
        if deterministic:
            action = masked_action_logits.argmax(dim=-1)
        else:
            action = action_dist.sample()

        direction_mask = self.get_direction_mask(int(action.item()))
        
        if not direction_mask.any():
            raise RuntimeError(f"No legal direction for selected action={action.item()}")
            
        masked_direction_logits = direction_logits.masked_fill(
            ~direction_mask,
            torch.finfo(direction_logits.dtype).min,
        )
        
        direction_dist = Categorical(logits=masked_direction_logits)
        
        if deterministic:
            direction = masked_direction_logits.argmax(dim=-1)
        else:
            direction = direction_dist.sample()

        action_log_prob = action_dist.log_prob(action)
        direction_log_prob = direction_dist.log_prob(direction)

        return {
            "action": action,
            "direction": direction,
            "action_log_prob": float(action_log_prob.item()),
            "direction_log_prob": float(direction_log_prob.item()),
            "action_mask": action_mask,
            "direction_mask": direction_mask,
        }

    def apply_action_choice(self, choice: dict):
        """Apply a previously sampled choice unless another agent interrupted it."""
        action = choice["action"]
        direction = choice["direction"]

        if self.done:
            # A rival occupied this agent's cell after both policies had made
            # their choices. Keep a terminal transition for the pending death
            # event, but do not advance time, burn energy, or alter the board.
            return (
                int(action.item()),
                int(direction.item()),
                choice["action_log_prob"],
                choice["direction_log_prob"],
                0.0,
                True,
                choice["direction_mask"],
            )

        self.pending_food_potential = self._food_distance_potential()
        self.previous_action = self.last_action
        self.last_direction = self.direction
        self.direction = int(direction.item())
        self.pre_action_position = (self.x, self.y)
        self.x_prev, self.y_prev = self.pre_action_position
        self.steps_since_last_food += 1
        self.steps += 1

        self.last_completion_reward = 0.0
        action_result = self.perform_action(action, direction)
        self.ate_last_step = bool(action_result.get("ate", False))
        self.blocked_last_step = (
            int(action.item()) == Action.MOVE.value and not bool(action_result.get("moved", False))
        )
        reward, reward_breakdown = self.calculate_rewards(action_result)
        terminal_reward = self._apply_terminal_limits()
        reward += terminal_reward
        reward_breakdown["terminal"] += terminal_reward
        
        self.last_reward = reward
        self.total_reward += reward
        
        for k, v in reward_breakdown.items():
            self.reward_ledger[k] += v

        return (
            int(action.item()),
            int(direction.item()),
            choice["action_log_prob"],
            choice["direction_log_prob"],
            reward,
            self.done,
            choice["direction_mask"],
        )

    def take_action(
        self,
        action_logits,
        direction_logits,
        *,
        action_mask: torch.Tensor = None,
        deterministic: bool = False,
    ):
        choice = self.sample_action_choice(
            action_logits,
            direction_logits,
            action_mask=action_mask,
            deterministic=deterministic,
        )
        return self.apply_action_choice(choice)

    def _apply_terminal_limits(self) -> float:
        delta = 0.0
        if self.energy <= 0 and not self.done:
            self.done = True
            self.died = True
        max_steps = int(getattr(Config, "MAX_EPISODE_STEPS", 0))
        if max_steps > 0 and self.steps >= max_steps and not self.done:
            self.done = True
            self.time_limit_reached = True
        incomplete_terminal = self.energy <= 0 or self.time_limit_reached
        if self.done and incomplete_terminal and not self.death_penalty_applied:
            initial_food = max(1, int(self.env.initial_food_count))
            remaining_fraction = min(
                1.0,
                max(0.0, float(self.env.food_remaining) / float(initial_food)),
            )
            terminal_scale = (
                self.TIME_LIMIT_MAX_PENALTY
                if self.time_limit_reached
                else self.ENERGY_DEATH_MAX_PENALTY
            )
            delta += terminal_scale * remaining_fraction
            self.death_penalty_applied = True
        return delta

    def finalize_environment_effects(self) -> float:
        """Finish autonomous effects and potential shaping for one transition."""
        delta = 0.0
        if self.pending_bouncer_hits > 0:
            hits = self.pending_bouncer_hits
            self.pending_bouncer_hits = 0
            self.collided_with_bouncing_obstacle = True
            self.energy -= self.ENERGY_PENALTY_BOUNCING_COLLISION * hits
            delta += self.BOUNCING_COLLISION_REWARD * hits
        if self.pending_snail_hits > 0:
            self.pending_snail_hits = 0
            self.killed_by_snail = True
            self.died = True
            self.done = True
            self.time_limit_reached = False
            snail_penalty = float(Config.IMMORTAL_SNAIL_DEATH_REWARD)
            delta += snail_penalty
            self.reward_ledger["terminal"] += snail_penalty
        if self.pending_agent_kill > 0:
            self.pending_agent_kill = 0
            self.killed_by_agent = True
            self.died = True
            self.done = True
            self.time_limit_reached = False
            kill_penalty = float(self.AGENT_KILL_DEATH_REWARD)
            delta += kill_penalty
            self.reward_ledger["terminal"] += kill_penalty
        terminal_delta = self._apply_terminal_limits()
        delta += terminal_delta
        self.reward_ledger["terminal"] += terminal_delta

        if self.pending_food_potential is not None:
            next_potential = self._food_distance_potential(terminal=self.done)
            shaping = (
                float(Config.GAMMA) * next_potential
                - self.pending_food_potential
            )
            self.pending_food_potential = None
            delta += shaping
            self.reward_ledger["distance_shaping"] += shaping

        self.last_reward += delta
        self.total_reward += delta
        return delta

    def _food_distance_potential(self, *, terminal: bool = False) -> float:
        """Potential whose discounted difference rewards progress toward food."""
        if terminal or not self.env.food_cells:
            return 0.0
        closest = min(
            abs(self.x - food_x) + abs(self.y - food_y)
            for food_x, food_y in self.env.food_cells
        )
        return -self.FOOD_DISTANCE_POTENTIAL_SCALE * float(closest)

    def perform_action(self, action, direction):
        """
        action, direction: 0-dim torch.Tensors from Categorical.sample()
        """
        a = int(action.item())
        d = int(direction.item())
        self.last_action = a
        if 0 <= a < len(self.action_counts):
            self.action_counts[a] += 1
        # Direction is recorded for every action, but it is meaningless for WAIT
        # (its mask pins index 0), so read the direction histogram against the
        # MOVE/GRAB/PLACE counts rather than the total.
        if a != Action.WAIT.value and 0 <= d < len(self.direction_counts):
            self.direction_counts[d] += 1

        if self.done:
            # Killed earlier in this same tick. The transition is still stored
            # so the terminal penalty from finalize_environment_effects has a
            # home, but the corpse takes no action on the board.
            return {
                'reward': 0.0, 'ledger_category': "movement", 'moved': False,
                'ate': False, 'moved_obstacle': False, 'failed_attempt': False,
            }

        if a == Action.MOVE.value:
            return self.handle_movement(d)
        if a == Action.WAIT.value:
            return self.handle_wait()
        return self.handle_obstacle_interaction(a, d)

    def choice_histogram(self) -> str:
        """Compact per-episode action/direction usage, e.g.
        Actions[move:71 grab:0 place:0 wait:57] Dirs[up:18 down:20 left:16 right:17]"""
        acts = " ".join(
            f"{name.lower()}:{self.action_counts[value]}"
            for name, value in ((a.name, a.value) for a in Action)
            if value < len(self.action_counts)
        )
        dirs = " ".join(
            f"{name.lower()}:{self.direction_counts[value]}"
            for name, value in ((d.name, d.value) for d in Direction)
            if value < len(self.direction_counts)
        )
        return f"Actions[{acts}] Dirs[{dirs}]"

    def handle_wait(self):
        """Hold position deliberately.

        A tick of world time passes - obstacles move, food flees, hunger advances -
        but the agent attempted nothing, so it carries no invalid-move penalty.
        It still burns a full metabolic tick; the STEP_REWARD time pressure also
        applies in calculate_rewards exactly as for any other turn.
        """
        return {
            'reward': 0.0,
            'ledger_category': "movement",
            'moved': False,
            'ate': False,
            'moved_obstacle': False,
            'failed_attempt': False,
        }

    # -------------------------------
    # Rewards aggregation
    # -------------------------------
    def calculate_rewards(self, action_result):
        # Breakdown dict to return
        breakdown = {k: 0.0 for k in self.reward_ledger.keys()}
        
        action_reward = action_result['reward']
        completion_reward = float(self.last_completion_reward)
        action_without_completion = action_reward - completion_reward
        if action_result.get("ate", False):
            breakdown["food"] += action_without_completion
        else:
            breakdown[action_result.get("ledger_category", "movement")] += (
                action_without_completion
            )
        breakdown["completion"] += completion_reward
        reward = action_reward
        
        step_rew = self.STEP_REWARD * self.steps if Config.USE_STEP_MULTIPLIER else self.STEP_REWARD
        reward += step_rew
        breakdown["step"] += step_rew

        if action_result['ate']:
            self.steps_since_last_food = 0
            
        if self.last_action == Action.MOVE.value and not action_result.get("moved", False):
            # Blocked streak penalty
            self.stalled_ticks += 1
            blocked_pen = self.REDUNDANT_MOVEMENT_REWARD * self.stalled_ticks
            reward += blocked_pen
            breakdown["blocked_streak"] += blocked_pen
        else:
            self.stalled_ticks = 0

        if (
            self.previous_action == Action.MOVE.value
            and self.last_action == Action.MOVE.value
            and self.last_direction == self.direction
            and self.direction_counter > 0
        ):
            reward += self.DIRECTION_MAINTAINED_REWARD / self.direction_counter

        reward += self.check_adjacent_food_not_eaten_last_position(action_result['ate'])

        # Base per-step energy burn. WAIT still pays for the passing tick, but at a
        # reduced rate because no movement was attempted. Every other outcome -
        # including a blocked or invalid move - pays full rate: the effort was spent
        # whether or not the agent actually got anywhere.
        burn = self.ENERGY_CONSUMPTION_RATE
        if self.last_action == Action.WAIT.value:
            burn *= self.ENERGY_WAIT_FRACTION
        self.energy -= burn
        return reward, breakdown

    def check_adjacent_food_not_eaten_last_position(self, ate):
        r, c = self.pre_action_position
        return self.ADJACENT_FOOD_NOT_EATEN_REWARD if self.env.get_adjacent_and_diagonal_food(r, c) and not ate else 0

    # -------------------------------
    # Movement & interactions
    # -------------------------------
    def handle_movement(self, direction):
        dx, dy = self.get_movement_delta(direction)
        new_x, new_y = self.x + dx, self.y + dy

        # Out-of-bounds check first so we can return a proper boundary penalty
        if not (0 <= new_x < self.env.size and 0 <= new_y < self.env.size):
            return self.handle_invalid_move(target_cell_value=None, out_of_bounds=True)

        target_cell_value = CellType(self.env.grid[new_x, new_y])

        if target_cell_value == CellType.PURSUER:
            # Contact is legal-but-lethal rather than an action-mask shortcut.
            self.env.queue_snail_hit(self)
            return {
                'reward': 0.0,
                'ledger_category': "movement",
                'moved': False,
                'ate': False,
                'moved_obstacle': False,
                'failed_attempt': False,
            }

        # Moveable obstacle: attempt to push via env
        if target_cell_value == CellType.MOVEABLE_OBSTACLE:
            return self.execute_move(new_x, new_y)  # env.move_agent() handles push success/fail

        # If it's a valid empty/food move, do it
        if self.is_valid_move(new_x, new_y):
            return self.execute_move(new_x, new_y)

        # Otherwise it's an invalid move into a specific non-moveable thing
        return self.handle_invalid_move(target_cell_value=target_cell_value, out_of_bounds=False)

    def is_valid_move(self, new_x, new_y):
        invalid_states = {
            CellType.OBSTACLE, CellType.GRABBABLE_OBSTACLE, CellType.AGENT,
            CellType.BOUNCING_OBSTACLE, CellType.LOOPING_OBSTACLE,
            CellType.COUNTER_LOOPING_OBSTACLE, CellType.WANDERING_OBSTACLE
        }
        if self.env.agent_predation_enabled:
            # An occupied cell is a legal target: entering it is the kill.
            invalid_states.discard(CellType.AGENT)
        return 0 <= new_x < self.env.size and 0 <= new_y < self.env.size and self.env.grid[new_x, new_y] not in invalid_states

    def execute_move(self, new_x, new_y):
        target_cell_value = CellType(self.env.grid[new_x, new_y])
        moved, ate, moved_obstacle, blocked_by_obstacle = self.env.move_agent(self.x, self.y, new_x, new_y)
        unvisited = (new_x, new_y) not in self.visited_cells

        if moved:
            # Successful move (empty/food OR successful push)
            self.update_agent_position(new_x, new_y, ate)
            self.visited_cells.add((self.x, self.y))
            if self.previous_action == Action.MOVE.value and self.direction == self.last_direction:
                self.direction_counter = (self.direction_counter or 0) + 1
            else:
                self.direction_counter = 1
            return {
                'reward': self.calculate_movement_reward(new_x, new_y, ate, unvisited, moved_obstacle, blocked_by_obstacle, target_cell_value, moved=True),
                'ledger_category': (
                    "predation" if target_cell_value == CellType.AGENT else "movement"
                ),
                'moved': True, 'ate': ate, 'moved_obstacle': moved_obstacle, 'failed_attempt': blocked_by_obstacle
            }

        # Movement failed (e.g., push attempt failed)
        return {
            'reward': self.calculate_movement_reward(new_x, new_y, ate, unvisited, moved_obstacle, blocked_by_obstacle, target_cell_value, moved=False),
            'ledger_category': "invalid_move",
            'moved': False, 'ate': False, 'moved_obstacle': moved_obstacle, 'failed_attempt': blocked_by_obstacle
        }

    def calculate_movement_reward(self, new_x, new_y, ate, unvisited, moved_obstacle, failed_attempt, grid_value, moved=False):
        reward = 0.0
    
        if ate:
            reward += self.calculate_dynamic_food_reward()
            self.direction_counter = 0
            self.num_grabbable_obstacle_interactions = 0
            self.num_placeable_obstacle_interactions = 0
            if self.env.food_remaining <= 0:
                if not uses_competitive_food_share_completion(self.env):
                    self.last_completion_reward = self.completion_reward()
                    reward += self.last_completion_reward
                self.done = True
            return reward
    
        if not ate and self.env.food_remaining <= 0:
            reward += self.NO_FOOD_LEFT_REWARD
            self.done = True
            return reward
    
        # --- pushing case ---
        if grid_value == CellType.MOVEABLE_OBSTACLE:
            # push outcome (and energy) lives here
            reward += self.check_moved_obstacle(moved_obstacle, failed_attempt)
            # if we actually moved into the obstacle's cell, also count movement shaping
            if moved_obstacle:
                reward += self.UNVISITED_EMPTY_REWARD if unvisited else self.EMPTY_REWARD
            return reward
    
        # --- normal movement shaping ---
        if grid_value == CellType.EMPTY:
            reward += self.UNVISITED_EMPTY_REWARD if unvisited else self.EMPTY_REWARD
        elif grid_value == CellType.AGENT:
            # Only a move that actually landed is a kill. A blocked attempt
            # reaches here too, and must not be paid.
            if moved and self.env.agent_predation_enabled:
                reward += self.AGENT_KILL_REWARD
                self.kills += 1
                self.energy = min(
                    self.max_energy,
                    self.energy
                    + self.energy_for_food * self.AGENT_KILL_ENERGY_FRACTION,
                )
            else:
                reward += self.OTHER_AGENT_REWARD
        elif grid_value == CellType.OBSTACLE:
            reward += self.OBSTACLE_REWARD
        elif grid_value == CellType.LOOPING_OBSTACLE:
            reward += self.LOOPING_OBSTACLE_REWARD
        elif grid_value == CellType.COUNTER_LOOPING_OBSTACLE:
            reward += self.COUNTER_LOOPING_OBSTACLE_REWARD
        elif grid_value == CellType.WANDERING_OBSTACLE:
            reward += self.WANDERING_OBSTACLE_REWARD
    
        return reward


    def handle_invalid_move(self, target_cell_value=None, out_of_bounds=False):
        """
        Assigns specific penalties for different invalid-move types.
        Returns the standard action_result dict.
        """
        if out_of_bounds:
            penalty = self.BOUNDARY_REWARD
        else:
            # Map specific invalid targets to small penalties
            mapping = {
                CellType.OBSTACLE: self.INVALID_STATIC_OBSTACLE_PENALTY,
                CellType.GRABBABLE_OBSTACLE: self.INVALID_GRABBABLE_OBSTACLE_PENALTY,
                CellType.AGENT: self.INVALID_OTHER_AGENT_PENALTY,
                CellType.BOUNCING_OBSTACLE: self.INVALID_DYNAMIC_OBSTACLE_PENALTY,
                CellType.LOOPING_OBSTACLE: self.INVALID_DYNAMIC_OBSTACLE_PENALTY,
                CellType.COUNTER_LOOPING_OBSTACLE: self.INVALID_DYNAMIC_OBSTACLE_PENALTY,
                CellType.WANDERING_OBSTACLE: self.INVALID_DYNAMIC_OBSTACLE_PENALTY,
            }
            penalty = mapping.get(target_cell_value, self.MOVEMENT_FAILURE_REWARD)

        return {
            'reward': penalty,
            'ledger_category': "invalid_move",
            'moved': False,
            'ate': False,
            'moved_obstacle': False,
            'failed_attempt': False
        }

    # -------------------------------
    # Obstacle/Inventory interactions
    # -------------------------------
    def check_moved_obstacle(self, moved_obstacle, failed_attempt):
        if moved_obstacle:
            self.num_moveable_obstacle_interactions += 1
            self.energy -= self.ENERGY_EXTRA_FOR_PUSH_SUCCESS
            return self.MOVEABLE_OBSTACLE_REWARD / float(self.num_moveable_obstacle_interactions)
    
        if failed_attempt:
            self.energy -= self.ENERGY_EXTRA_FOR_PUSH_FAIL
            return self.MOVEABLE_OBSTACLE_FAIL_REWARD
    
        return 0.0

        
    def handle_obstacle_interaction(self, action, direction):
        grabbable_obstacles = self.env.get_adjacent_grabbable_obstacles(self.x, self.y)
        reward = 0.0
        if direction < 0 or direction >= len(CARDINAL_DELTAS):
            return {'reward': self.GRABBABLE_OBSTACLE_FAIL_REWARD, 'moved': False,
                    'ate': False, 'moved_obstacle': False, 'failed_attempt': True}
        act_direction = CARDINAL_DELTAS[direction]

        if action == Action.GRAB.value:
            interaction_reward = self.handle_grab_action(grabbable_obstacles, act_direction)
            success = (interaction_reward != self.GRABBABLE_OBSTACLE_FAIL_REWARD)
        elif action == Action.PLACE.value:
            interaction_reward = self.handle_place_action(act_direction)
            success = (interaction_reward != self.PLACEABLE_OBSTACLE_FAIL_REWARD)
        else:
            interaction_reward = self.GRABBABLE_OBSTACLE_FAIL_REWARD
            success = False

        reward += interaction_reward
        return {
            'reward': reward,
            'ledger_category': "interaction_success" if success else "interaction_fail",
            'moved': False, 'ate': False, 'moved_obstacle': False, 'failed_attempt': not success
        }

    def handle_grab_action(self, grabbable_obstacles, act_direction):
        if grabbable_obstacles:
            for obstacle in grabbable_obstacles:
                if obstacle == (self.x + act_direction[0], self.y + act_direction[1]) and len(self.inventory) < Config.MAX_INVENTORY:
                    self.inventory.append(CellType.GRABBABLE_OBSTACLE)
                    obstacle_x, obstacle_y = self.x + act_direction[0], self.y + act_direction[1]
                    self.env._update_cell(obstacle_x, obstacle_y, CellType.EMPTY)
                    self.env._refresh_push_reservations()
                    self.num_grabbable_obstacle_interactions += 1
                    self.energy -= (self.ENERGY_CONSUMPTION_RATE * self.ENERGY_EXTRA_FOR_GRAB_SUCCESS)
                    return self.GRABBABLE_OBSTACLE_REWARD if self.num_grabbable_obstacle_interactions == 1 else self.GRABBABLE_OBSTACLE_REWARD * (self.num_grabbable_obstacle_interactions + 1) * -1

        self.energy -= (self.ENERGY_CONSUMPTION_RATE * self.ENERGY_EXTRA_FOR_GRAB_FAIL)
        return self.GRABBABLE_OBSTACLE_FAIL_REWARD

    def handle_place_action(self, act_direction):
        if self.inventory:
            place_x, place_y = self.x + act_direction[0], self.y + act_direction[1]
            if self.env.is_within_bounds(place_x, place_y) and (place_x, place_y) in self.env.empty_cells and self.env.grid[place_x, place_y] != CellType.MOVEABLE_OBSTACLE:
                self.env._update_cell(place_x, place_y, CellType.GRABBABLE_OBSTACLE)
                self.env._refresh_push_reservations()
                self.env.placed_obstacles.add((place_x, place_y))
                self.inventory.pop(0)
                self.num_placeable_obstacle_interactions += 1
                self.energy -= (self.ENERGY_CONSUMPTION_RATE * self.ENERGY_EXTRA_FOR_PLACE_SUCCESS)
                return self.PLACEABLE_OBSTACLE_REWARD if self.num_placeable_obstacle_interactions == 1 else self.PLACEABLE_OBSTACLE_REWARD * (self.num_placeable_obstacle_interactions + 1) * -1

        self.energy -= (self.ENERGY_CONSUMPTION_RATE * self.ENERGY_EXTRA_FOR_PLACE_FAIL)
        return self.PLACEABLE_OBSTACLE_FAIL_REWARD

    # -------------------------------
    # Utilities
    # -------------------------------
    def update_agent_position(self, new_x, new_y, ate):
        # env.agents already updated by env.move_agent(...)
        self.x, self.y = new_x, new_y
        if ate:
            self.food_eaten += 1
            self.energy += self.energy_for_food

    def get_movement_delta(self, direction):
        return CARDINAL_DELTAS[direction] if 0 <= direction < len(CARDINAL_DELTAS) else (0, 0)

    def calculate_dynamic_food_reward(self):
        multiplier = 1.0
        if Config.USE_CONTEXT_FOOD_CHAIN_BONUS and self.food_eaten > 1:
            horizon = max(1, int(self.context_length))
            # The eating action itself advances steps_since_last_food once, so
            # subtract it: consecutive eating actions have zero intervening steps.
            intervening_steps = max(0, int(self.steps_since_last_food) - 1)
            decay = min(1.0, intervening_steps / float(horizon))
            minimum = float(Config.FOOD_CHAIN_MIN_MULTIPLIER)
            maximum = float(Config.FOOD_CHAIN_MAX_MULTIPLIER)
            multiplier = maximum - (maximum - minimum) * decay
            multiplier = max(minimum, min(maximum, multiplier))
        self.food_reward_multipliers.append(multiplier)
        return self.FOOD_REWARD * multiplier

    def completion_reward(self) -> float:
        """Scale full-board credit with the amount of work in this arena."""
        per_food = float(
            getattr(Config, "COMPLETION_REWARD_PER_INITIAL_FOOD", 0.0)
        )
        if per_food > 0.0:
            return per_food * float(self.env.initial_food_count)
        return float(self.COMPLETION_REWARD)

class TransitionStorage:
    """Stores transitions with optional truncation bootstrap support."""
    def __init__(self):
        self.transitions = []

    def store_transition(
        self,
        state,
        action,
        direction,
        action_log_prob,
        direction_log_prob,
        reward,
        value,
        done,
        truncated: bool = False,
        bootstrap_value: torch.Tensor | None = None,
    ):
        """
        Store one transition. `bootstrap_value` is only needed for the *last*
        transition of a truncated rollout segment. It must be V(s_{t+1})
        computed from the next state.
        """
        # Unpack and move to CPU immediately
        # Unpack and move to CPU immediately
        if len(state) == 5:
            view_seq, state_vec, pos_seq, action_mask, direction_mask = state
            action_mask_cpu = action_mask.detach().cpu()
            direction_mask_cpu = direction_mask.detach().cpu()
        elif len(state) == 4:
            view_seq, state_vec, pos_seq, action_mask = state
            action_mask_cpu = action_mask.detach().cpu()
            direction_mask_cpu = None
        else:
            view_seq, state_vec, pos_seq = state
            action_mask_cpu = None
            direction_mask_cpu = None
            
        # Grid cells are small integer enums. Byte-packing the 1,024-transition
        # ceiling saves about 12 MiB; the model converts them to long immediately
        # before embedding, so uint8 is lossless here.
        view_cpu = view_seq.detach().to(device="cpu", dtype=torch.uint8)
        state_cpu = state_vec.detach().cpu()
        pos_cpu = pos_seq.detach().cpu()
        
        value_cpu = (value if isinstance(value, torch.Tensor)
                     else torch.tensor(value, dtype=torch.float32)).detach().cpu()
        action_log_prob_cpu = (
            float(action_log_prob.detach().cpu().item())
            if isinstance(action_log_prob, torch.Tensor)
            else float(action_log_prob)
        )
        direction_log_prob_cpu = (
            float(direction_log_prob.detach().cpu().item())
            if isinstance(direction_log_prob, torch.Tensor)
            else float(direction_log_prob)
        )

        if truncated and bootstrap_value is None:
            raise ValueError("If truncated=True, bootstrap_value (V(s_{t+1})) is required.")

        if bootstrap_value is not None:
            bootstrap_cpu = (bootstrap_value if isinstance(bootstrap_value, torch.Tensor)
                             else torch.tensor(bootstrap_value, dtype=torch.float32)).detach().cpu()
            self.transitions.append((
                (view_cpu, state_cpu, pos_cpu, action_mask_cpu, direction_mask_cpu),
                action,
                direction,
                action_log_prob_cpu,
                direction_log_prob_cpu,
                reward,
                value_cpu,
                bool(done),
                True,               # truncated
                bootstrap_cpu       # V(s_{t+1})
            ))
        else:
            self.transitions.append((
                (view_cpu, state_cpu, pos_cpu, action_mask_cpu, direction_mask_cpu),
                action,
                direction,
                action_log_prob_cpu,
                direction_log_prob_cpu,
                reward,
                value_cpu,
                bool(done)
            ))

    def mark_last_as_truncated(self, bootstrap_value: torch.Tensor):
        """
        Upgrade the most recent transition (if it exists and is NOT terminal)
        to include truncated=True and the provided `bootstrap_value` (V(s_{t+1})).
        """
        if not self.transitions:
            return
        last = self.transitions[-1]
        done_flag = bool(last[7])  # 'done' is always at slot 7 in both layouts
        if done_flag:
            return

        bootstrap_cpu = (bootstrap_value if isinstance(bootstrap_value, torch.Tensor)
                         else torch.tensor(bootstrap_value, dtype=torch.float32)).detach().cpu()

        if len(last) == 8:
            # (state, action, direction, a_logp, d_logp, reward, value, done)
            rebuilt = last + (True, bootstrap_cpu)
        else:
            # (state, action, direction, a_logp, d_logp, reward, value, done, truncated, bootstrap)
            rebuilt = last[:9] + (bootstrap_cpu,)
        self.transitions[-1] = rebuilt

    def mark_last_as_time_limit_truncated(
        self, bootstrap_value: torch.Tensor
    ) -> None:
        """Convert a hard-cap terminal into a bootstrapped PPO truncation."""
        if not self.transitions:
            raise RuntimeError("cannot truncate an empty transition buffer")
        last = self.transitions[-1]
        bootstrap_cpu = (
            bootstrap_value
            if isinstance(bootstrap_value, torch.Tensor)
            else torch.tensor(bootstrap_value, dtype=torch.float32)
        ).detach().cpu()
        # The environment loop still ends through agent.done. PPO must instead
        # see done=False plus a segment boundary and V(s_{t+1}).
        self.transitions[-1] = last[:7] + (
            False,
            True,
            bootstrap_cpu,
        )

    def add_reward_to_last_transition(self, reward_delta: float) -> None:
        """Retroactively credit an episode-level result before PPO consumes it."""
        if not self.transitions:
            raise RuntimeError("cannot credit an empty transition buffer")
        last = self.transitions[-1]
        self.transitions[-1] = (
            *last[:5],
            float(last[5]) + float(reward_delta),
            *last[6:],
        )

    def clear_memory(self):
        self.transitions.clear()

