"""Environment simulation classes for NavGrid."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

import numpy as np

from .config import Config
from .constants import (
    CARDINAL_DELTAS,
    Action,
    CellType,
    Direction,
    EpisodeVariant,
    FoodMovementMode,
    ObstacleSetupChoice,
)

if TYPE_CHECKING:
    from .agent import Agent

@dataclass(frozen=True)
class CurriculumScenario:
    """One replayable environment regime."""
    label: str
    difficulty: int
    food_tick: bool
    food_mode: str
    obstacle_choice: int
    food_tick_speed: int = 4
    weight: float = 1.0




def uses_competitive_food_share_completion(env: "GridEnvironment") -> bool:
    """Return whether this episode awards its clear bonus by food ownership."""
    return bool(
        getattr(Config, "COMPETITIVE_FOOD_SHARE_COMPLETION", False)
    ) and env.episode_variant is EpisodeVariant.DUAL_AGENT



class SpatialHash:
    def __init__(self, cell_size=5):
        self.cell_size = cell_size
        self.buckets = {}

    def insert(self, r, c):
        bucket = (r // self.cell_size, c // self.cell_size)
        if bucket not in self.buckets:
            self.buckets[bucket] = []
        self.buckets[bucket].append((r, c))

    def iter_nearby(self, r, c, radius):
        min_b_r = (r - radius) // self.cell_size
        max_b_r = (r + radius) // self.cell_size
        min_b_c = (c - radius) // self.cell_size
        max_b_c = (c + radius) // self.cell_size

        for b_r in range(min_b_r, max_b_r + 1):
            for b_c in range(min_b_c, max_b_c + 1):
                bucket = (b_r, b_c)
                if bucket in self.buckets:
                    for item in self.buckets[bucket]:
                        yield item

class FoodSwarmController:
    """Discrete food movement controller with random-walk and boid-style policies.

    Food is still represented as ordinary CellType.FOOD cells in the grid. This
    controller only adds enough per-food movement memory to make flocking possible
    without changing the model input vocabulary, the reward path, or the agent code.
    """

    CARDINAL_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    CANDIDATE_DELTAS = [(0, 0)] + CARDINAL_DELTAS

    def __init__(self):
        self.velocities = {}
        self.spatial_hash = None

    def reset(self):
        """Clear per-food movement memory at episode reset."""
        self.velocities.clear()
        self.spatial_hash = None

    def sync_food(self, food_cells: set[tuple[int, int]]):
        """Prune eaten food and seed velocities for newly spawned/moved food."""
        live_cells = set(food_cells)
        for cell in list(self.velocities.keys()):
            if cell not in live_cells:
                self.velocities.pop(cell, None)

        for cell in live_cells:
            if cell not in self.velocities:
                self.velocities[cell] = random.choice(self.CARDINAL_DELTAS)

    def update_random(self, env):
        """Preserve the original random adjacent food movement behavior."""
        self.sync_food(env.food_cells)
        for r, c in list(env.food_cells):
            if (r, c) not in env.food_cells:
                continue

            empty_adjacent = [
                cell for cell in env._get_adjacent_cells(r, c, {CellType.EMPTY})
                if env.is_food_destination_allowed(cell)
            ]
            if empty_adjacent:
                new_r, new_c = random.choice(empty_adjacent)
                self._record_velocity((r, c), (new_r, new_c))
                env._move_item(r, c, new_r, new_c, CellType.FOOD)

        self.sync_food(env.food_cells)

    def update_boids(self, env):
        """Move food as a discrete flock that avoids agents and preserves collision safety."""
        self.sync_food(env.food_cells)
        if not env.food_cells:
            return

        original_food_cells = set(env.food_cells)
        self.spatial_hash = SpatialHash(cell_size=max(5, Config.FOOD_BOID_NEIGHBOR_RADIUS))
        for r, c in original_food_cells:
            self.spatial_hash.insert(r, c)

        planned_moves = {}
        reserved_targets = set()

        # Sort for deterministic conflict resolution; the tiny jitter still breaks ties.
        for cell in sorted(original_food_cells):
            if cell not in env.food_cells:
                continue

            target = self._choose_boid_target(env, cell, original_food_cells, reserved_targets)
            if target != cell:
                planned_moves[cell] = target
                reserved_targets.add(target)

        for old_cell, new_cell in planned_moves.items():
            if old_cell not in env.food_cells:
                continue
            if not env.is_within_bounds(new_cell[0], new_cell[1]):
                continue
            if CellType(env.grid[new_cell[0], new_cell[1]]) != CellType.EMPTY:
                continue

            self._record_velocity(old_cell, new_cell)
            env._move_item(old_cell[0], old_cell[1], new_cell[0], new_cell[1], CellType.FOOD)

        self.sync_food(env.food_cells)

    def _record_velocity(self, old_cell: tuple[int, int], new_cell: tuple[int, int]):
        dr = new_cell[0] - old_cell[0]
        dc = new_cell[1] - old_cell[1]
        self.velocities.pop(old_cell, None)
        self.velocities[new_cell] = (dr, dc)

    def _choose_boid_target(self, env, cell: tuple[int, int], original_food_cells: set[tuple[int, int]], reserved_targets: set[tuple[int, int]]) -> tuple[int, int]:
        candidates = []
        for dr, dc in self.CANDIDATE_DELTAS:
            candidate = (cell[0] + dr, cell[1] + dc)
            if self._is_valid_candidate(env, cell, candidate, reserved_targets):
                candidates.append(candidate)

        if not candidates:
            return cell

        neighbors_neighbor = self._local_food_neighbors(cell, original_food_cells, Config.FOOD_BOID_NEIGHBOR_RADIUS)
        neighbors_sep = [n for n in neighbors_neighbor if self._chebyshev(cell, n) <= Config.FOOD_BOID_SEPARATION_RADIUS]

        # --- Precompute Boids Constants ---
        cohesion_center = None
        if len(neighbors_neighbor) >= 2:
            center_r = sum(r for r, _ in neighbors_neighbor) / len(neighbors_neighbor)
            center_c = sum(c for _, c in neighbors_neighbor) / len(neighbors_neighbor)
            cohesion_center = (center_r, center_c)

        alignment_avg_vel = None
        if neighbors_neighbor:
            avg_dr, avg_dc, used = 0.0, 0.0, 0
            for neighbor in neighbors_neighbor:
                velocity = self.velocities.get(neighbor)
                if velocity is not None:
                    avg_dr += velocity[0]
                    avg_dc += velocity[1]
                    used += 1
            if used > 0:
                avg_dr /= used
                avg_dc /= used
                avg_norm = math.sqrt(avg_dr * avg_dr + avg_dc * avg_dc)
                if avg_norm > 1e-6:
                    alignment_avg_vel = (avg_dr / avg_norm, avg_dc / avg_norm)

        separation_old_pressure = 0.0
        if neighbors_sep:
            for neighbor in neighbors_sep:
                old_dist = max(0.25, self._euclidean(cell, neighbor))
                separation_old_pressure += 1.0 / ((old_dist + 0.35) ** 2)

        best_score = -float("inf")
        best_target = cell
        for candidate in candidates:
            score = self._score_candidate(env, cell, candidate, cohesion_center, separation_old_pressure, alignment_avg_vel, neighbors_sep)
            if score > best_score:
                best_score = score
                best_target = candidate

        return best_target

    def _is_valid_candidate(self, env, cell: tuple[int, int], candidate: tuple[int, int], reserved_targets: set[tuple[int, int]]) -> bool:
        if candidate == cell:
            return True
        if candidate in reserved_targets:
            return False
        if candidate != cell and not env.is_food_destination_allowed(candidate):
            return False
        if not env.is_within_bounds(candidate[0], candidate[1]):
            return False
        return CellType(env.grid[candidate[0], candidate[1]]) == CellType.EMPTY

    def _score_candidate(self, env, cell: tuple[int, int], candidate: tuple[int, int], cohesion_center, separation_old_pressure: float, alignment_avg_vel, neighbors_sep: list[tuple[int, int]]) -> float:
        move_vec = (candidate[0] - cell[0], candidate[1] - cell[1])
        score = random.uniform(-Config.FOOD_BOID_RANDOM_WEIGHT, Config.FOOD_BOID_RANDOM_WEIGHT)

        if move_vec == (0, 0):
            # Staying is legal, but movement should win when a safe useful move exists.
            score -= 0.10

        score += self._agent_avoidance_score(env, cell, candidate)
        score += self._cohesion_score(cell, candidate, cohesion_center)
        score += self._separation_score(candidate, separation_old_pressure, neighbors_sep)
        score += self._alignment_score(move_vec, alignment_avg_vel)
        score += self._momentum_score(move_vec, cell)
        score += self._wall_score(env, cell, candidate)
        return score

    def _agent_avoidance_score(self, env, cell: tuple[int, int], candidate: tuple[int, int]) -> float:
        if not env.agents:
            return 0.0

        old_pressure = 0.0
        new_pressure = 0.0
        for agent_cell in env.agents:
            old_pressure += self._agent_pressure(self._manhattan(cell, agent_cell))
            new_pressure += self._agent_pressure(self._manhattan(candidate, agent_cell))

        # Repel food until it reaches the configured safety radius, then stop
        # rewarding extra distance. The earlier version kept rewarding every step
        # away from the agent, which naturally drove the whole flock into edges.
        return Config.FOOD_BOID_AGENT_WEIGHT * (old_pressure - new_pressure)

    def _cohesion_score(self, cell: tuple[int, int], candidate: tuple[int, int], center) -> float:
        if center is None:
            return 0.0

        target_radius = max(0.0, Config.FOOD_BOID_COHESION_TARGET_RADIUS)
        old_dist = self._euclidean(cell, center)
        new_dist = self._euclidean(candidate, center)

        # Cohesion is intentionally band-limited: food should remain a loose herd,
        # not collapse onto the center of mass and form a single sticky blob.
        old_error = abs(old_dist - target_radius)
        new_error = abs(new_dist - target_radius)
        return Config.FOOD_BOID_COHESION_WEIGHT * (old_error - new_error)

    def _separation_score(self, candidate: tuple[int, int], old_pressure_sum: float, neighbors: list[tuple[int, int]]) -> float:
        if not neighbors:
            return 0.0

        new_pressure_sum = 0.0
        for neighbor in neighbors:
            new_dist = max(0.25, self._euclidean(candidate, neighbor))
            # Inverse-square pressure makes adjacent/diagonal crowding expensive
            # while still allowing a loose herd at a few cells of spacing.
            new_pressure_sum += 1.0 / ((new_dist + 0.35) ** 2)

        return Config.FOOD_BOID_SEPARATION_WEIGHT * (old_pressure_sum - new_pressure_sum)

    def _alignment_score(self, move_vec: tuple[int, int], avg_vel_normalized) -> float:
        if move_vec == (0, 0) or avg_vel_normalized is None:
            return 0.0

        dot = (move_vec[0] * avg_vel_normalized[0] + move_vec[1] * avg_vel_normalized[1])
        return Config.FOOD_BOID_ALIGNMENT_WEIGHT * dot

    def _momentum_score(self, move_vec: tuple[int, int], cell: tuple[int, int]) -> float:
        if move_vec == (0, 0):
            return 0.0

        velocity = self.velocities.get(cell)
        if velocity is None:
            return 0.0

        return Config.FOOD_BOID_MOMENTUM_WEIGHT * (move_vec[0] * velocity[0] + move_vec[1] * velocity[1])

    def _wall_score(self, env, cell: tuple[int, int], candidate: tuple[int, int]) -> float:
        old_pressure = self._wall_pressure(env, cell)
        new_pressure = self._wall_pressure(env, candidate)

        # Reward movement out of the border pressure field and lightly penalize
        # ending inside it. This breaks edge/corner lockups even when cohesion or
        # agent avoidance is trying to squeeze the herd against a wall.
        return Config.FOOD_BOID_WALL_WEIGHT * ((old_pressure - new_pressure) - 0.15 * new_pressure)

    def _local_food_neighbors(self, cell: tuple[int, int], food_cells: set[tuple[int, int]], radius: int) -> list[tuple[int, int]]:
        if self.spatial_hash is None:
            return []
        
        r, c = cell
        return [
            neighbor for neighbor in self.spatial_hash.iter_nearby(r, c, radius)
            if neighbor != cell and max(abs(r - neighbor[0]), abs(c - neighbor[1])) <= radius
        ]

    @staticmethod
    def _agent_pressure(distance: float) -> float:
        radius = max(0.0, float(Config.FOOD_BOID_AGENT_AVOID_RADIUS))
        if radius <= 0.0 or distance >= radius:
            return 0.0
        normalized = (radius - distance) / radius
        return normalized ** 3

    @staticmethod
    def _manhattan(a, b) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _chebyshev(a, b) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    @staticmethod
    def _euclidean(a, b) -> float:
        return math.dist(a, b)

    @staticmethod
    def _edge_distance(env, cell: tuple[int, int]) -> int:
        return min(cell[0], cell[1], env.size - 1 - cell[0], env.size - 1 - cell[1])

    @staticmethod
    def _wall_pressure(env, cell: tuple[int, int]) -> float:
        margin = max(0, int(Config.FOOD_BOID_WALL_MARGIN))
        if margin <= 0:
            return 0.0

        r, c = cell
        distances = (r, c, env.size - 1 - r, env.size - 1 - c)
        pressure = 0.0
        for distance in distances:
            if distance <= margin:
                pressure += ((margin + 1 - distance) / (margin + 1)) ** 2
        return pressure


class GridEnvironment:
    """Grid world simulation managing items, agents, and obstacle dynamics."""

    # Read Config dynamically; do not cache this at import time.
    ADJACENT_DIRECTIONS = list(CARDINAL_DELTAS)
    DIAGONAL_DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    ALL_DIRECTIONS = ADJACENT_DIRECTIONS + DIAGONAL_DIRECTIONS

    def __init__(self, idx, size=None):
        """
        Initializes the environment instance.

        Args:
            idx (int): Unique index for this environment instance.
            size (int, optional): Size of the grid (width/height). Defaults to Config.ENVIRONMENT_SIZE.
        """
        self.idx = idx
        # Use instance variable for size, defaulting to Config setting
        self.size = size if size is not None else Config.ENVIRONMENT_SIZE
        self.curriculum_scenario_idx: Optional[int] = None
        self.curriculum_scenario: Optional[CurriculumScenario] = None
        self.episode_variant = EpisodeVariant.STANDARD
        self.agent_predation_enabled = False
        self.food_tick_enabled = bool(Config.FOOD_TICK)

        self.initial_food_count = 0 # Initial food count for this episode

        # Instance variables for configuration determined during reset
        self.num_food = 0
        self.num_obstacles = 0
        self.num_moveable_obstacles = 0
        self.num_grabbable_obstacles = 0
        self.num_bouncing_obstacles = 0
        # Instance variable for food tick speed
        self.food_tick_speed = Config.FOOD_TICK_SPEED
        self.food_movement_mode = FoodMovementMode.resolve_episode_mode(Config.FOOD_MOVEMENT_MODE)
        self.food_swarm = FoodSwarmController()

        self.world_type = 0 # Stores the ObstacleSetupChoice for this instance
        self.agents = set()  # occupied agent coordinates
        self.agent_objects: dict[int, "Agent"] = {}
        self.agent_by_position: dict[tuple[int, int], "Agent"] = {}
        self.new_grid_generated = True
        self.episode_cleared = False

        # Initialize grid and state sets
        self.grid = None
        self.empty_cells = set()
        self.food_cells = set()
        self.obstacle_cells = set()
        self.moveable_obstacle_cells = set()
        self.grabbable_obstacle_cells = set()
        self.bouncing_obstacle_objects = set() # Stores BouncingObstacle objects
        self.snail_pursuers: dict[int, "SnailPursuer"] = {}
        self.snail_requested = False
        self.snail_spawn_failed = False
        self.pursuer_cells = set()
        self._generation_sealed_cells: Optional[set[tuple[int, int]]] = None
        self.moved_obstacles = set() # Track recently moved obstacles (for visualization?)
        self.placed_obstacles = set() # Track recently placed obstacles (for visualization?)
        self.food_remaining = 0
        self.reset()

    def configure_curriculum_scenario(
        self,
        scenario_idx: int,
        scenario: CurriculumScenario,
    ) -> None:
        """Assign the scenario consumed by this environment's next reset."""
        self.curriculum_scenario_idx = int(scenario_idx)
        self.curriculum_scenario = scenario
        self.food_tick_enabled = bool(scenario.food_tick)

    def configure_episode_variant(self, variant: EpisodeVariant | str) -> None:
        """Apply one episode condition without leaking it into global rules."""
        self.episode_variant = EpisodeVariant(variant)
        self.agent_predation_enabled = (
            self.episode_variant is EpisodeVariant.DUAL_AGENT
        )

    def reset(self):
        """Generate a fresh, strictly valid episode without recursive resets."""
        attempts = max(1, int(getattr(Config, "REACHABILITY_REGEN_ATTEMPTS", 20)))
        last_problem = "unknown generation failure"
        for attempt in range(1, attempts + 1):
            self._initialize_grid()
            self._determine_item_counts()
            try:
                self._distribute_items()
                ok, problems = self._perform_sanity_checks()
            except ValueError as e:
                ok = False
                problems = [str(e)]
            
            if ok:
                self.agents.clear()
                self.agent_objects.clear()
                self.agent_by_position.clear()
                self.new_grid_generated = True
                self.episode_cleared = self.food_remaining <= 0
                return
            last_problem = "; ".join(problems)
            if Config.DEBUG:
                raise AssertionError(
                    f"Env {self.idx} generation failed on attempt {attempt}: {last_problem}"
                )
            print(
                f"Warning: Env {self.idx} rejected generated world "
                f"({attempt}/{attempts}): {last_problem}"
            )
        raise RuntimeError(
            f"Env {self.idx} could not generate a valid episode after {attempts} attempts: "
            f"{last_problem}"
        )

    def make_render_snapshot(self, tick: Optional[int] = None, consume_events: bool = True) -> EnvRenderSnapshot:
        """Copy a thread-safe render frame and optionally consume one-frame events."""
        snapshot = EnvRenderSnapshot(
            grid=self.grid.copy(),
            moved_obstacles=tuple(self.moved_obstacles),
            placed_obstacles=tuple(self.placed_obstacles),
            tick=tick,
        )
        if consume_events:
            self.moved_obstacles.clear()
            self.placed_obstacles.clear()
        return snapshot

    def _initialize_grid(self):
        """
        Initialize the grid to EMPTY and reset all cell/item tracking sets.
        """
        self.grid = np.full((self.size, self.size), CellType.EMPTY, dtype=int) # Use dtype=int
        self.empty_cells = set((r, c) for r in range(self.size) for c in range(self.size))
        self.food_cells = set()
        self.obstacle_cells = set()
        self.moveable_obstacle_cells = set()
        self.grabbable_obstacle_cells = set()
        self.bouncing_obstacle_objects = set()
        self.snail_pursuers = {}
        self.snail_requested = False
        self.snail_spawn_failed = False
        self.pursuer_cells = set()
        self._generation_sealed_cells = None
        self.moved_obstacles = set()
        self.placed_obstacles = set()
        self.reserved_for_push = set()
        self.food_remaining = 0
        self.episode_cleared = False
        self.competitive_completion_awarded = False
        self.episode_variant = EpisodeVariant.STANDARD
        self.agent_predation_enabled = False
        self.agents = set()
        self.agent_objects = {}
        self.agent_by_position = {}
        if hasattr(self, "food_swarm"):
            self.food_swarm.reset()
        # self.agents is cleared in reset()

    def _determine_item_counts(self):
        """
        Determines the number of food and various obstacles for this episode
        based on Config settings, ensuring a minimum percentage of empty cells
        and prioritizing the intended food count during capping.
        Sets instance variables: self.num_food, self.num_obstacles, etc.
        Also sets self.food_tick_speed for the instance.
        """
        total_cells = self.size * self.size
        required_agent_space = Config.NUM_AGENTS if hasattr(Config, 'NUM_AGENTS') else 1
        max_items = math.floor(total_cells * (1.0 - float(Config.MIN_EMPTY_PERCENTAGE)))
        max_items = min(max_items, total_cells - required_agent_space)
        max_items = max(0, max_items)

        # --- Determine Initial Food Count (using original logic where specified) ---
        initial_num_food = 0
        if Config.VARIABLE_FOOD_COUNT:
            # Use random but ensure at least 1
            initial_num_food = random.randint(1, max(1, total_cells // 10))
        elif Config.SCALE_FOOD_COUNT:
            # Use scaling, ensure at least 1 (matches original if size >= 4)
            initial_num_food = max(1, total_cells // 10)
        else:
            # Use fixed count, ensure at least 1 if positive
            initial_num_food = max(0, Config.SET_FOOD_COUNT) # Allow 0 if config is 0
            if Config.SET_FOOD_COUNT > 0:
                 initial_num_food = max(1, initial_num_food) # Ensure 1 if intended > 0

        # Cap initial food if it *alone* exceeds max_items
        initial_num_food = min(initial_num_food, max_items)
        # Ensure at least 1 food if it was intended and max_items allows it
        if initial_num_food == 0 and (Config.VARIABLE_FOOD_COUNT or Config.SCALE_FOOD_COUNT or Config.SET_FOOD_COUNT > 0) and max_items >= 1:
            initial_num_food = 1
            print(f"Info: Env {self.idx} - Adjusted initial food count to 1 due to capping.")

        final_num_food = initial_num_food # Start with the potentially capped initial food count

        # --- Determine Per-Environment Dynamics ---
        scenario = self.curriculum_scenario
        if scenario is not None:
            self.food_tick_enabled = bool(scenario.food_tick)
            self.food_tick_speed = (
                max(1, int(scenario.food_tick_speed))
                if self.food_tick_enabled
                else 0
            )
            self.food_movement_mode = FoodMovementMode.resolve_episode_mode(
                scenario.food_mode
            )
        elif Config.USE_RANDOM_FOOD_TICK:
            self.food_tick_enabled = bool(Config.FOOD_TICK)
            self.food_tick_speed = random.randint(0, Config.MAX_FOOD_TICK_SPEED)
        else:
            self.food_tick_enabled = bool(Config.FOOD_TICK)
            self.food_tick_speed = Config.FOOD_TICK_SPEED
        if scenario is None:
            self.food_movement_mode = FoodMovementMode.resolve_episode_mode(
                Config.FOOD_MOVEMENT_MODE
            )

        # --- Determine Initial Obstacle Counts ---
        total_obstacle_count = total_cells // 3
        break_down_count = max(1, total_obstacle_count // 4)

        if scenario is None and Config.VARIABLE_OBSTACLE_COUNT:
            choice = random.choice(list(ObstacleSetupChoice))
        else:
            try:
                choice = ObstacleSetupChoice(
                    scenario.obstacle_choice
                    if scenario is not None
                    else Config.OBSTACLE_CHOICE
                )
            except ValueError:
                print(f"Warning: Invalid Config.OBSTACLE_CHOICE ({Config.OBSTACLE_CHOICE}). Defaulting to MIXED_ALL_TYPES.")
                choice = ObstacleSetupChoice.MIXED_ALL_TYPES
        self.world_type = choice

        num_obstacles = 0; num_moveable = 0; num_grabbable = 0; num_bouncing = 0
        if choice in [ObstacleSetupChoice.ONLY_STATIC_LARGE, ObstacleSetupChoice.ONLY_STATIC_BOUNCE_MED]: num_obstacles = total_obstacle_count
        if choice in [ObstacleSetupChoice.ONLY_MOVEABLE_LARGE, ObstacleSetupChoice.ONLY_MOVEABLE_BOUNCE_MED]: num_moveable = total_obstacle_count
        if choice in [ObstacleSetupChoice.ONLY_GRABBABLE_LARGE, ObstacleSetupChoice.ONLY_GRABBABLE_BOUNCE_MED]: num_grabbable = total_obstacle_count
        if choice in [ObstacleSetupChoice.MIXED_SMALL_NO_BOUNCE, ObstacleSetupChoice.MIXED_SMALL_BOUNCE_LOW, ObstacleSetupChoice.MIXED_SMALL_BOUNCE_MED]: num_obstacles = num_moveable = num_grabbable = break_down_count
        if choice == ObstacleSetupChoice.EMPTY_NO_OBSTACLES: pass
        if choice == ObstacleSetupChoice.MIXED_RANDOM_SMALL_NO_BOUNCE: num_obstacles, num_moveable, num_grabbable = random.randint(0, break_down_count), random.randint(0, break_down_count), random.randint(0, break_down_count)
        if choice in [ObstacleSetupChoice.STATIC_MOVEABLE_HALF, ObstacleSetupChoice.MOVEABLE_GRABBABLE_HALF, ObstacleSetupChoice.STATIC_GRABBABLE_HALF]:
            half_count = total_obstacle_count // 2
            if choice == ObstacleSetupChoice.STATIC_MOVEABLE_HALF: num_obstacles, num_moveable = half_count, half_count
            elif choice == ObstacleSetupChoice.MOVEABLE_GRABBABLE_HALF: num_moveable, num_grabbable = half_count, half_count
            elif choice == ObstacleSetupChoice.STATIC_GRABBABLE_HALF: num_obstacles, num_grabbable = half_count, half_count
        if choice in [ObstacleSetupChoice.MOVEABLE_GRABBABLE_BOUNCE_MED, ObstacleSetupChoice.STATIC_GRABBABLE_BOUNCE_MED, ObstacleSetupChoice.STATIC_MOVEABLE_BOUNCE_MED]:
            num_obstacles = break_down_count if choice in [ObstacleSetupChoice.STATIC_GRABBABLE_BOUNCE_MED, ObstacleSetupChoice.STATIC_MOVEABLE_BOUNCE_MED] else 0
            num_moveable = break_down_count if choice in [ObstacleSetupChoice.MOVEABLE_GRABBABLE_BOUNCE_MED, ObstacleSetupChoice.STATIC_MOVEABLE_BOUNCE_MED] else 0
            num_grabbable = break_down_count if choice in [ObstacleSetupChoice.MOVEABLE_GRABBABLE_BOUNCE_MED, ObstacleSetupChoice.STATIC_GRABBABLE_BOUNCE_MED] else 0
            num_bouncing = break_down_count // 3
        if choice == ObstacleSetupChoice.MIXED_ALL_TYPES: num_obstacles, num_moveable, num_grabbable, num_bouncing = random.randint(0, break_down_count), random.randint(0, break_down_count), random.randint(0, break_down_count), random.randint(0, break_down_count // 2)
        if choice in [ ObstacleSetupChoice.MIXED_SMALL_BOUNCE_MED, ObstacleSetupChoice.MOVEABLE_GRABBABLE_BOUNCE_MED, ObstacleSetupChoice.STATIC_GRABBABLE_BOUNCE_MED, ObstacleSetupChoice.STATIC_MOVEABLE_BOUNCE_MED, ObstacleSetupChoice.ONLY_STATIC_BOUNCE_MED, ObstacleSetupChoice.ONLY_MOVEABLE_BOUNCE_MED, ObstacleSetupChoice.ONLY_GRABBABLE_BOUNCE_MED, ObstacleSetupChoice.MIXED_ALL_TYPES ]: num_bouncing = max(num_bouncing, break_down_count // 4)

        # --- Apply Obstacle Capping (Prioritizing Food) ---
        max_allowed_obstacles = max(0, max_items - final_num_food)
        current_obstacles_total = num_obstacles + num_moveable + num_grabbable + num_bouncing

        if current_obstacles_total > max_allowed_obstacles:
            obstacles_to_remove = current_obstacles_total - max_allowed_obstacles
            print(f"Warning: Env {self.idx} - Initial obstacle count ({current_obstacles_total}) exceeds max allowed ({max_allowed_obstacles}) after placing food. Reducing {obstacles_to_remove} obstacles.")

            obstacles_removed_count = 0
            if obstacles_to_remove > 0 and current_obstacles_total > 0:
                 # Build pool for proportional removal
                 pool = []
                 if num_obstacles > 0: pool.extend([CellType.OBSTACLE] * num_obstacles)
                 if num_moveable > 0: pool.extend([CellType.MOVEABLE_OBSTACLE] * num_moveable)
                 if num_grabbable > 0: pool.extend([CellType.GRABBABLE_OBSTACLE] * num_grabbable)
                 if num_bouncing > 0: pool.extend([CellType.BOUNCING_OBSTACLE] * num_bouncing)

                 random.shuffle(pool)
                 # Remove items from the pool and decrement counts
                 for i in range(obstacles_to_remove):
                     if not pool: break
                     item_to_remove = pool.pop()
                     if item_to_remove == CellType.OBSTACLE: num_obstacles -= 1
                     elif item_to_remove == CellType.MOVEABLE_OBSTACLE: num_moveable -= 1
                     elif item_to_remove == CellType.GRABBABLE_OBSTACLE: num_grabbable -= 1
                     elif item_to_remove == CellType.BOUNCING_OBSTACLE: num_bouncing -= 1
                     obstacles_removed_count += 1

                 # Sanity check after removal
                 final_obstacle_total = num_obstacles + num_moveable + num_grabbable + num_bouncing
                 if final_obstacle_total != max_allowed_obstacles:
                      print(f"Warning: Env {self.idx} - Obstacle capping mismatch. Target: {max_allowed_obstacles}, Final: {final_obstacle_total}")

        # --- Assign Final Counts to Instance Variables ---
        self.num_food = final_num_food
        self.initial_food_count = final_num_food # Store the initial food count for this episode
        self.num_obstacles = max(0, num_obstacles)
        self.num_moveable_obstacles = max(0, num_moveable)
        self.num_grabbable_obstacles = max(0, num_grabbable)
        self.num_bouncing_obstacles = max(0, num_bouncing)

        # Final check (optional debug)
        final_total_items = self.num_food + self.num_obstacles + self.num_moveable_obstacles + self.num_grabbable_obstacles + self.num_bouncing_obstacles
        if final_total_items > max_items:
             print(f"ERROR: Env {self.idx} - Final item count ({final_total_items}) STILL exceeds max ({max_items}) after capping!")
        # print(f"Env {self.idx} Final Counts: Food={self.num_food}, Obs={self.num_obstacles}, Move={self.num_moveable_obstacles}, Grab={self.num_grabbable_obstacles}, Bounce={self.num_bouncing_obstacles}")

    def _distribute_items(self):
        """
        Place obstacles first, then distribute food and spawns using a local 
        anti-sealing invariant rather than global reachability carving.
        """
        self._distribute_randomly(CellType.OBSTACLE, self.num_obstacles)
        self._distribute_randomly(CellType.MOVEABLE_OBSTACLE, self.num_moveable_obstacles)
        self._distribute_randomly(CellType.GRABBABLE_OBSTACLE, self.num_grabbable_obstacles)
        self._distribute_bouncing_obstacles(self.num_bouncing_obstacles)

        # Static obstacles are the only permanent generation blockers. Build
        # this once and reuse it for food placement, spawn selection, and
        # sanity checks instead of rescanning every cell for each phase.
        sealed_cells = self._refresh_generation_sealed_cells()

        # Recalculate empty_cells accurately after all obstacle placements
        empty_value = CellType.EMPTY.value
        self.empty_cells = set(
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.grid[r, c] == empty_value
        )

        food_candidates = [
            cell
            for cell in self.empty_cells
            if cell not in sealed_cells
        ]

        if len(food_candidates) < self.num_food:
            raise ValueError(f"Env {self.idx} has {len(food_candidates)} non-sealed valid empty cells, but {self.num_food} food requested.")

        sampled_food = random.sample(food_candidates, self.num_food)
        for cell in sampled_food:
            self._update_cell(cell[0], cell[1], CellType.FOOD)

        if Config.DEBUG:
            assert self.food_remaining == self.num_food, (
                f"Env {self.idx} - food_remaining ({self.food_remaining}) != num_food ({self.num_food}) after placement")
        elif self.food_remaining != self.num_food:
            print(f"Warning: Env {self.idx} - food_remaining ({self.food_remaining}) != num_food ({self.num_food}) after placement")


    def _distribute_randomly(self, item_type: CellType, count: int):
        """
        Helper method to distribute a specific item type randomly onto empty grid cells.
        Updates the grid and relevant tracking sets.

        Args:
            item_type (CellType): The type of item to distribute.
            count (int): Number of items to distribute.
        """
        if count <= 0:
            return
        # Ensure we don't try to place more items than available empty cells
        actual_count = min(count, len(self.empty_cells))
        if actual_count < count:
            print(f"Warning: Requested {count} of {item_type.name}, but only {actual_count} empty cells available.")

        if actual_count == 0:
            return # No empty cells left

        # Sample empty cells without replacement
        cells_to_place = random.sample(list(self.empty_cells), actual_count)

        for cell in cells_to_place:
            self._update_cell(cell[0], cell[1], item_type)

    def _distribute_bouncing_obstacles(self, count: int):
        """
        Distributes bouncing obstacles, creating BouncingObstacle objects.
        Updates grid, empty_cells set, and bouncing_obstacle_objects set.

        Args:
            count (int): Number of bouncing obstacles to distribute.
        """
        if count <= 0:
            return

        actual_count = min(count, len(self.empty_cells))
        if actual_count < count:
            print(f"Warning: Requested {count} bouncing obstacles, but only {actual_count} empty cells available.")

        if actual_count == 0:
            return

        directions = ["up", "down", "left", "right"]
        cells_to_place = random.sample(list(self.empty_cells), actual_count)

        for cell in cells_to_place:
            direction = random.choice(directions)
            # Create the BouncingObstacle object
            # Ensure BouncingObstacle class is defined elsewhere and takes (x, y, dir, size)
            try:
                bouncing_obstacle = BouncingObstacle(cell[0], cell[1], direction, self.size)
                self.bouncing_obstacle_objects.add(bouncing_obstacle) # Store the object
                self._update_cell(cell[0], cell[1], bouncing_obstacle.grid_value)
            except NameError:
                print("Error: BouncingObstacle class not defined. Skipping bouncing obstacle placement.")
                break # Stop trying if the class isn't available
            except Exception as e:
                print(f"Error creating BouncingObstacle at {cell}: {e}")
        self.num_bouncing_obstacles = len(self.bouncing_obstacle_objects)
 
    # All grid values that block agent traversal at generation time.
    # Moveable/grabbable are included: the guarantee must hold without pushes/grabs.
    OBSTACLE_GRID_VALUES = {
        CellType.OBSTACLE,
        CellType.MOVEABLE_OBSTACLE,
        CellType.GRABBABLE_OBSTACLE,
        CellType.BOUNCING_OBSTACLE,
        CellType.LOOPING_OBSTACLE,
        CellType.COUNTER_LOOPING_OBSTACLE,
        CellType.WANDERING_OBSTACLE,
    }

    PERMANENT_GENERATION_BLOCKERS = {
        CellType.OBSTACLE,
    }

    def _permanent_blocker_count(self, r: int, c: int) -> int:
        blocked = 0
        for dr, dc in self.ADJACENT_DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not self.is_within_bounds(nr, nc):
                blocked += 1
            elif CellType(self.grid[nr, nc]) in self.PERMANENT_GENERATION_BLOCKERS:
                blocked += 1
        return blocked

    def _is_locally_permanently_sealed(self, r: int, c: int) -> bool:
        return (r, c) in self._get_generation_sealed_cells()

    def _refresh_generation_sealed_cells(self) -> set[tuple[int, int]]:
        """Cache cells enclosed on all four sides by walls/static obstacles."""
        blockers = (self.grid == CellType.OBSTACLE.value).astype(np.uint8)
        padded = np.pad(blockers, 1, mode="constant", constant_values=1)
        blocker_count = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )
        self._generation_sealed_cells = set(
            map(tuple, np.argwhere(blocker_count >= 4))
        )
        return self._generation_sealed_cells

    def _get_generation_sealed_cells(self) -> set[tuple[int, int]]:
        if self._generation_sealed_cells is None:
            return self._refresh_generation_sealed_cells()
        return self._generation_sealed_cells

    def _refresh_push_reservations(self) -> set[tuple[int, int]]:
        """Recompute currently useful push destinations for moving-food avoidance."""
        reserved: set[tuple[int, int]] = set()
        for obstacle_r, obstacle_c in self.moveable_obstacle_cells:
            for dr, dc in self.ADJACENT_DIRECTIONS:
                agent_r, agent_c = obstacle_r - dr, obstacle_c - dc
                dest_r, dest_c = obstacle_r + dr, obstacle_c + dc
                if not (self.is_within_bounds(agent_r, agent_c) and self.is_within_bounds(dest_r, dest_c)):
                    continue
                agent_side = CellType(self.grid[agent_r, agent_c])
                dest_type = CellType(self.grid[dest_r, dest_c])
                if agent_side in (CellType.EMPTY, CellType.FOOD, CellType.AGENT) and dest_type == CellType.EMPTY:
                    reserved.add((dest_r, dest_c))
        self.reserved_for_push = reserved
        return reserved

    def get_spawnable_cells(self, allow_fallback: bool = False) -> set:
        """Return valid spawn cells using local anti-sealing invariant."""
        spawnable = self.empty_cells - self._get_generation_sealed_cells()

        if spawnable:
            return spawnable

        if allow_fallback:
            return set()

        return set()

    def _update_cell(self, r: int, c: int, new_type: CellType):
        """
        Unified method to modify a cell's type on the grid.
        Automatically handles adding to and discarding from tracking sets.
        """
        old_type = CellType(self.grid[r, c])
        if old_type == new_type:
            return

        if (
            old_type in self.PERMANENT_GENERATION_BLOCKERS
            or new_type in self.PERMANENT_GENERATION_BLOCKERS
        ):
            self._generation_sealed_cells = None

        # Remove from old set
        if old_type == CellType.EMPTY:
            self.empty_cells.discard((r, c))
        elif old_type == CellType.FOOD:
            self.food_cells.discard((r, c))
            self.food_remaining = max(0, self.food_remaining - 1)
            if self.food_remaining == 0 and self.initial_food_count > 0:
                self.episode_cleared = True
        elif old_type == CellType.OBSTACLE:
            self.obstacle_cells.discard((r, c))
        elif old_type == CellType.MOVEABLE_OBSTACLE:
            self.moveable_obstacle_cells.discard((r, c))
            self.moved_obstacles.discard((r, c))
        elif old_type == CellType.GRABBABLE_OBSTACLE:
            self.grabbable_obstacle_cells.discard((r, c))
            self.placed_obstacles.discard((r, c))
        elif old_type == CellType.AGENT:
            self.agents.discard((r, c))
        elif old_type == CellType.PURSUER:
            self.pursuer_cells.discard((r, c))

        # Set new value
        self.grid[r, c] = new_type

        # Add to new set
        if new_type == CellType.EMPTY:
            self.empty_cells.add((r, c))
        elif new_type == CellType.FOOD:
            self.food_cells.add((r, c))
            self.food_remaining += 1
            self.episode_cleared = False
        elif new_type == CellType.OBSTACLE:
            self.obstacle_cells.add((r, c))
        elif new_type == CellType.MOVEABLE_OBSTACLE:
            self.moveable_obstacle_cells.add((r, c))
        elif new_type == CellType.GRABBABLE_OBSTACLE:
            self.grabbable_obstacle_cells.add((r, c))
        elif new_type == CellType.AGENT:
            self.agents.add((r, c))
        elif new_type == CellType.PURSUER:
            self.pursuer_cells.add((r, c))

    def _perform_sanity_checks(self) -> tuple[bool, list[str]]:
        """Independently verify reset bookkeeping and local sealing invariant."""
        problems: list[str] = []

        for r, c in self.food_cells:
            if self._is_locally_permanently_sealed(r, c):
                problems.append(
                    f"food at {(r, c)} is locally sealed by permanent blockers"
                )

        spawnable = self.get_spawnable_cells(allow_fallback=False)
        if not spawnable:
            problems.append("no valid non-sealed spawn cell")

        reserved_food = self.food_cells & self.reserved_for_push
        if reserved_food:
            problems.append(f"{len(reserved_food)} food cells occupy push destinations")
        if self.food_remaining != len(self.food_cells):
            problems.append("food_remaining disagrees with food_cells")
        if self.num_bouncing_obstacles != len(self.bouncing_obstacle_objects):
            problems.append("bouncing obstacle count disagrees with object set")
        expected_pursuers = set(
            map(tuple, np.argwhere(self.grid == CellType.PURSUER.value))
        )
        if expected_pursuers != self.pursuer_cells:
            problems.append("pursuer_cells is out of sync")
        object_pursuers = (
            {snail.position for snail in self.snail_pursuers.values()}
            if self.snail_pursuers
            else set()
        )
        if expected_pursuers != object_pursuers:
            problems.append("snail object disagrees with grid")

        if Config.DEBUG:
            # Diagnostics: informational only
            traversable = self.empty_cells | self.food_cells
            largest_comp = set()
            if traversable:
                remaining = set(traversable)
                while remaining:
                    seed = remaining.pop()
                    comp = {seed}
                    frontier = [seed]
                    while frontier:
                        r, c = frontier.pop()
                        for dr, dc in self.ADJACENT_DIRECTIONS:
                            neighbor = (r + dr, c + dc)
                            if neighbor in remaining:
                                remaining.discard(neighbor)
                                comp.add(neighbor)
                                frontier.append(neighbor)
                    if len(comp) > len(largest_comp):
                        largest_comp = comp
            
            food_outside = len(self.food_cells - largest_comp)
            sealed_empty = len(
                self.empty_cells & self._get_generation_sealed_cells()
            )
            
            print(
                f"[GEN]\n"
                f"food_requested={self.num_food}\n"
                f"food_placed={len(self.food_cells)}\n"
                f"valid_food_candidates={len(self.empty_cells) - sealed_empty}\n"
                f"valid_spawn_candidates={len(spawnable)}\n"
                f"sealed_empty_candidates={sealed_empty}\n"
                f"food_outside_largest_empty_component={food_outside}"
            )

        return not problems, problems

    def validate_state(self, raise_on_error: bool = True) -> list[str]:
        """Check grid/set/object invariants after arbitrary dynamic interaction."""
        problems: list[str] = []
        expected_empty = set(map(tuple, np.argwhere(self.grid == CellType.EMPTY.value)))
        expected_food = set(map(tuple, np.argwhere(self.grid == CellType.FOOD.value)))
        expected_agents = set(map(tuple, np.argwhere(self.grid == CellType.AGENT.value)))
        if expected_empty != self.empty_cells:
            problems.append("empty_cells is out of sync")
        if expected_food != self.food_cells:
            problems.append("food_cells is out of sync")
        if expected_agents != self.agents:
            problems.append("agents coordinate set is out of sync")
        expected_pursuers = set(
            map(tuple, np.argwhere(self.grid == CellType.PURSUER.value))
        )
        if expected_pursuers != self.pursuer_cells:
            problems.append("pursuer_cells is out of sync")
        object_pursuers = (
            {snail.position for snail in self.snail_pursuers.values()}
            if self.snail_pursuers
            else set()
        )
        if expected_pursuers != object_pursuers:
            problems.append("snail object disagrees with grid")
        object_positions = {(o.x, o.y) for o in self.bouncing_obstacle_objects}
        grid_dynamic = {
            tuple(x) for x in np.argwhere(np.isin(self.grid, [
                CellType.BOUNCING_OBSTACLE.value,
                CellType.LOOPING_OBSTACLE.value,
                CellType.COUNTER_LOOPING_OBSTACLE.value,
                CellType.WANDERING_OBSTACLE.value,
            ]))
        }
        if object_positions != grid_dynamic:
            problems.append("bouncing obstacle objects disagree with grid")
        if self.food_remaining != len(self.food_cells):
            problems.append("food_remaining is out of sync")
        if set(self.agent_by_position) != self.agents:
            problems.append("agent_by_position is out of sync")
        if problems and raise_on_error:
            raise AssertionError(f"Env {self.idx} invariant failure: {'; '.join(problems)}")
        return problems

    def register_agent(self, agent: "Agent", cell: tuple[int, int]) -> None:
        """Place/rebind an agent with stable object identity."""
        old_positions = [pos for pos, obj in self.agent_by_position.items() if obj is agent]
        for old in old_positions:
            self.agent_by_position.pop(old, None)
            if old in self.agents and CellType(self.grid[old]) == CellType.AGENT:
                self._update_cell(old[0], old[1], CellType.EMPTY)
        if CellType(self.grid[cell]) != CellType.EMPTY:
            raise ValueError(f"Cannot spawn agent {agent.agent_idx} on {CellType(self.grid[cell]).name} at {cell}")
        self.agent_objects[agent.agent_idx] = agent
        self.agent_by_position[cell] = agent
        self._update_cell(cell[0], cell[1], CellType.AGENT)

    def get_agent_at(self, r: int, c: int) -> Optional["Agent"]:
        return self.agent_by_position.get((r, c))

    def queue_bouncer_hit(self, r: int, c: int, obstacle_id: int) -> None:
        """Attach a collision to the actual agent object immediately."""
        agent = self.get_agent_at(r, c)
        if agent is not None and not agent.done:
            agent.pending_bouncer_hits += 1
            agent.last_bouncer_id = obstacle_id

    def queue_snail_hit(self, agent: "Agent") -> None:
        """Attach lethal pursuer contact to the action's current transition."""
        if not agent.done:
            agent.pending_snail_hits += 1

    def queue_agent_kill(self, agent: "Agent") -> None:
        """Attach a lethal predation contact to the victim's current transition.

        Unlike the pursuer, which moves only after every agent has acted, a
        predation kill lands in the middle of the agent loop. The victim is
        retired here so it cannot act again from a cell the attacker now owns.
        """
        if not agent.done:
            agent.pending_agent_kill += 1
            agent.done = True

    def configure_snail_for_episode(self) -> None:
        """Place pursuers only for an explicitly selected snail episode."""
        self.snail_pursuers = {}
        self.pursuer_cells.clear()
        self.snail_spawn_failed = False
        self.snail_requested = bool(
            getattr(Config, "USE_IMMORTAL_SNAIL", False)
        ) and self.episode_variant is EpisodeVariant.IMMORTAL_SNAIL
        if not self.snail_requested:
            return
        live_agents = [agent for agent in self.agent_objects.values() if not agent.done]
        if not live_agents:
            raise RuntimeError("immortal-snail trainer requires at least one live agent")
        # One pursuer per resident, each anchored on the agent it hunts. Cells
        # inside another resident's minimum distance are refused, and each snail
        # occupies its cell before the next one picks, so they never stack.
        for agent in live_agents:
            snail = SnailPursuer.spawn(
                self,
                agent,
                min_distance=int(Config.IMMORTAL_SNAIL_MIN_SPAWN_DISTANCE),
                max_distance=int(Config.IMMORTAL_SNAIL_MAX_SPAWN_DISTANCE),
                move_interval=int(Config.IMMORTAL_SNAIL_MOVE_INTERVAL),
                others=[other for other in live_agents if other is not agent],
            )
            if snail is None:
                self.snail_spawn_failed = True
                continue
            self.snail_pursuers[int(agent.agent_idx)] = snail

    def get_view(self, center_r: int, center_c: int, agent_r: int, agent_c: int, view_size: int) -> np.ndarray:
        """Return a fixed-size discrete observation without silent top-left cropping."""
        if view_size <= 0:
            raise ValueError("view_size must be positive")

        if Config.USE_GLOBAL_STATE:
            processed = self._process_view_for_agents(self.grid, agent_r, agent_c, use_global=True)
            if self.size == view_size:
                return processed
            if self.size < view_size:
                out = np.full((view_size, view_size), CellType.BOUNDARY.value, dtype=int)
                off_r = (view_size - self.size) // 2
                off_c = (view_size - self.size) // 2
                out[off_r:off_r + self.size, off_c:off_c + self.size] = processed
                return out

            policy = str(getattr(Config, "GLOBAL_VIEW_OVERSIZE_POLICY", "nearest")).lower()
            if policy == "error":
                raise ValueError(
                    f"Global grid {self.size}x{self.size} exceeds model view {view_size}x{view_size}"
                )
            if policy == "agent_centered":
                center_r, center_c = agent_r, agent_c
            elif policy == "nearest":
                row_idx = np.rint(np.linspace(0, self.size - 1, view_size)).astype(np.int64)
                col_idx = np.rint(np.linspace(0, self.size - 1, view_size)).astype(np.int64)
                out = processed[np.ix_(row_idx, col_idx)].copy()
                mapped_r = int(np.argmin(np.abs(row_idx - agent_r)))
                mapped_c = int(np.argmin(np.abs(col_idx - agent_c)))
                out[out == CellType.AGENT.value] = CellType.OTHER_AGENT.value
                out[mapped_r, mapped_c] = CellType.AGENT.value
                return out
            else:
                raise ValueError(f"Unknown GLOBAL_VIEW_OVERSIZE_POLICY={policy!r}")

        half = view_size // 2

        wanted_r0 = center_r - half
        wanted_c0 = center_c - half
        wanted_r1 = wanted_r0 + view_size
        wanted_c1 = wanted_c0 + view_size

        src_r0 = max(0, wanted_r0)
        src_c0 = max(0, wanted_c0)
        src_r1 = min(self.size, wanted_r1)
        src_c1 = min(self.size, wanted_c1)

        dst_r0 = src_r0 - wanted_r0
        dst_c0 = src_c0 - wanted_c0

        out = np.full(
            (view_size, view_size),
            CellType.BOUNDARY.value,
            dtype=int,
        )

        height = src_r1 - src_r0
        width = src_c1 - src_c0

        if height > 0 and width > 0:
            out[
                dst_r0:dst_r0 + height,
                dst_c0:dst_c0 + width,
            ] = self.grid[src_r0:src_r1, src_c0:src_c1]
            
        return self._process_view_for_agents(out, agent_r, agent_c, use_global=False, view_r0=wanted_r0, view_c0=wanted_c0)

    def _process_view_for_agents(self, view_array: np.ndarray, agent_r: int, agent_c: int, use_global: bool, view_r0: int = 0, view_c0: int = 0) -> np.ndarray:
        """Distinguish the requesting agent from all other agents."""
        processed = view_array.copy()
        if use_global:
            for r, c in self.agents:
                if self.is_within_bounds(r, c):
                    processed[r, c] = (
                        CellType.AGENT.value if (r, c) == (agent_r, agent_c)
                        else CellType.OTHER_AGENT.value
                    )
            return processed

        processed[processed == CellType.AGENT.value] = CellType.OTHER_AGENT.value
        agent_dst_r = agent_r - view_r0
        agent_dst_c = agent_c - view_c0
        if 0 <= agent_dst_r < processed.shape[0] and 0 <= agent_dst_c < processed.shape[1]:
            processed[agent_dst_r, agent_dst_c] = CellType.AGENT.value
        return processed

    def move_agent(self, old_r: int, old_c: int, new_r: int, new_c: int) -> tuple[bool, bool, bool, bool]:
        """
        Attempts to move an agent and updates the environment state.

        Args:
            old_r, old_c (int): Current coordinates of the agent.
            new_r, new_c (int): Target coordinates to move to.

        Returns:
            tuple[bool, bool, bool, bool]: (success, ate_food, moved_obstacle, blocked_by_obstacle)
        """
        if not self.is_within_bounds(new_r, new_c):
            # Movement out of bounds is considered failure, not blocked
            return False, False, False, False

        target_cell_type = CellType(self.grid[new_r, new_c]) # Get type before potential modification

        return self._handle_movement(old_r, old_c, new_r, new_c, target_cell_type)

    def _handle_movement(self, old_r: int, old_c: int, new_r: int, new_c: int, target_type: CellType) -> tuple[bool, bool, bool, bool]:
        """
        Internal handler for movement logic based on the target cell type.
        Updates grid and sets upon successful movement.

        Returns:
            tuple[bool, bool, bool, bool]: (success, ate_food, moved_obstacle, blocked_by_obstacle)
        """
        if target_type == CellType.PURSUER:
            agent = self.get_agent_at(old_r, old_c)
            if agent is not None:
                self.queue_snail_hit(agent)
            return False, False, False, False

        if target_type == CellType.AGENT and self.agent_predation_enabled:
            victim = self.get_agent_at(new_r, new_c)
            attacker = self.get_agent_at(old_r, old_c)
            if victim is not None and victim is not attacker and not victim.done:
                self.queue_agent_kill(victim)
                self._perform_move(old_r, old_c, new_r, new_c, False)
                return True, False, False, False
            return False, False, False, True

        if target_type == CellType.EMPTY or target_type == CellType.FOOD:
            ate = (target_type == CellType.FOOD)
            self._perform_move(old_r, old_c, new_r, new_c, ate)
            return True, ate, False, False # Success, ate?, moved_obstacle=False, blocked=False

        elif target_type == CellType.MOVEABLE_OBSTACLE:
            return self._handle_moveable_obstacle(old_r, old_c, new_r, new_c)

        # Blocked by other obstacle types (static, grabbable, bouncing, looping, etc.) or other agents
        # Note: is_valid_move in Agent class should prevent attempts to move into these,
        # but this provides safety.
        return False, False, False, True # Failed, not_ate, not_moved_obstacle, blocked=True

    def _perform_move(self, old_r: int, old_c: int, new_r: int, new_c: int, ate_food: bool):
        """Execute a simple move while preserving stable agent identity."""
        agent = self.agent_by_position.pop((old_r, old_c), None)
        self._update_cell(new_r, new_c, CellType.AGENT)
        self._update_cell(old_r, old_c, CellType.EMPTY)
        if agent is not None:
            self.agent_by_position[(new_r, new_c)] = agent

    def _handle_moveable_obstacle(self, agent_r: int, agent_c: int, obstacle_r: int, obstacle_c: int) -> tuple[bool, bool, bool, bool]:
        """
        Handles the logic when an agent tries to move into a moveable obstacle.
        Attempts to push the obstacle. Updates grid and sets if successful.

        Returns:
            tuple[bool, bool, bool, bool]: (success, ate_food=False, moved_obstacle, blocked_by_obstacle)
        """
        # Calculate direction of push and the cell behind the obstacle
        dr, dc = obstacle_r - agent_r, obstacle_c - agent_c
        behind_r, behind_c = obstacle_r + dr, obstacle_c + dc

        # Check if the cell behind is within bounds and empty
        if self.is_within_bounds(behind_r, behind_c) and self.grid[behind_r, behind_c] == CellType.EMPTY:
            # --- Push successful ---
            agent = self.agent_by_position.pop((agent_r, agent_c), None)
            self._update_cell(behind_r, behind_c, CellType.MOVEABLE_OBSTACLE)
            self._update_cell(obstacle_r, obstacle_c, CellType.AGENT)
            self._update_cell(agent_r, agent_c, CellType.EMPTY)
            if agent is not None:
                self.agent_by_position[(obstacle_r, obstacle_c)] = agent
            self.moved_obstacles.add((behind_r, behind_c))
            self._refresh_push_reservations()
            # No post-action reachability repair: a bad push is allowed to sabotage the episode.
            return True, False, True, False
        else:
            # --- Push failed (blocked behind) ---
            return False, False, False, True # Failed, not_ate, moved_obstacle=False, blocked=True

    def is_food_destination_allowed(self, cell: tuple[int, int]) -> bool:
        if not bool(getattr(Config, "PRESERVE_PUSH_LANES_FROM_MOVING_FOOD", True)):
            return True
        return cell not in self.reserved_for_push

    def update_food(self, current_tick: int):
        """Advance food while preserving currently usable push destinations."""
        if not self.food_tick_enabled or self.food_tick_speed <= 0:
            return
        self._refresh_push_reservations()
        if current_tick % self.food_tick_speed == 0:
            if self.food_movement_mode == FoodMovementMode.BOIDS:
                self.food_swarm.update_boids(self)
            else:
                self.food_swarm.update_random(self)

    def build_non_empty_cells(self) -> set[tuple[int, int]]:
        s = set()
        s |= self.food_cells
        s |= self.obstacle_cells
        s |= self.moveable_obstacle_cells
        s |= self.grabbable_obstacle_cells
        s |= self.agents.copy()
        s |= self.pursuer_cells
        s |= {(o.x, o.y) for o in self.bouncing_obstacle_objects}
        return s

    def update_bouncing_obstacles(self, current_tick: int):
        """Advance mobile obstacles. Temporary trapping causes a stall, never a poof."""
        if not self.bouncing_obstacle_objects:
            return
        for obstacle in tuple(self.bouncing_obstacle_objects):
            obstacle.dead = False  # revive legacy objects saved with the old transient-death flag
            try:
                obstacle.update(self, current_tick)
            except Exception as exc:
                obstacle.stalled_ticks += 1
                print(
                    f"Warning: bouncing obstacle #{obstacle._id} stalled after error at "
                    f"({obstacle.x},{obstacle.y}): {exc}"
                )
                # Keep object/grid identity intact rather than deleting it.
                if self.is_within_bounds(obstacle.x, obstacle.y):
                    current = CellType(self.grid[obstacle.x, obstacle.y])
                    if current == CellType.EMPTY:
                        self._update_cell(obstacle.x, obstacle.y, obstacle.grid_value)
        self.num_bouncing_obstacles = len(self.bouncing_obstacle_objects)

    def update_snail_pursuer(self, current_tick: int) -> None:
        """Advance every lethal pursuer toward the resident it hunts."""
        if not self.snail_pursuers:
            return
        for agent in self.agent_objects.values():
            snail = self.snail_pursuers.get(int(agent.agent_idx))
            if snail is None or agent.done or agent.pending_snail_hits:
                continue
            snail.update(self, agent, current_tick)

    def _move_item(self, old_r: int, old_c: int, new_r: int, new_c: int, item_type: CellType) -> bool:
        """Move an item transactionally; return whether the move committed."""
        if CellType(self.grid[old_r, old_c]) != item_type:
            return False
        if CellType(self.grid[new_r, new_c]) != CellType.EMPTY:
            return False
        self._update_cell(new_r, new_c, item_type)
        self._update_cell(old_r, old_c, CellType.EMPTY)
        return True

    def is_within_bounds(self, r: int, c: int) -> bool:
        """ Checks if coordinates are within the grid boundaries. """
        return 0 <= r < self.size and 0 <= c < self.size

    def _get_adjacent_cells(self, r: int, c: int, target_types: set[CellType], directions: list[tuple]=ADJACENT_DIRECTIONS) -> list[tuple]:
        """
        Gets adjacent or diagonal cells matching specified types.

        Args:
            r (int): Row coordinate.
            c (int): Col coordinate.
            target_types (set[CellType]): A set of CellType enums to search for.
            directions (list[tuple], optional): List of (dr, dc) tuples.
                                                 Defaults to ADJACENT_DIRECTIONS.
                                                 Use ALL_DIRECTIONS for diagonals too.

        Returns:
            list[tuple]: List of (row, col) tuples of matching adjacent cells.
        """
        adjacent_cells = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self.is_within_bounds(nr, nc) and CellType(self.grid[nr, nc]) in target_types:
                adjacent_cells.append((nr, nc))
        return adjacent_cells

    def get_adjacent_empty_cells(self, r: int, c: int) -> list[tuple]:
        """ Gets adjacent empty cells. """
        return self._get_adjacent_cells(r, c, {CellType.EMPTY})

    def get_adjacent_food(self, r: int, c: int) -> list[tuple]:
        """ Gets adjacent food cells. """
        return self._get_adjacent_cells(r, c, {CellType.FOOD})

    def get_adjacent_and_diagonal_food(self, r: int, c: int) -> list[tuple]:
         """ Gets adjacent and diagonal food cells. """
         return self._get_adjacent_cells(r, c, {CellType.FOOD}, directions=self.ALL_DIRECTIONS)

    def get_adjacent_grabbable_obstacles(self, r: int, c: int) -> list[tuple]:
         """ Gets adjacent grabbable obstacle cells. """
         return self._get_adjacent_cells(r, c, {CellType.GRABBABLE_OBSTACLE})

    def get_empty_cells(self) -> set[tuple]:
         """ Returns a copy of the set of empty cells. """
         return self.empty_cells.copy()

    def update_all_sets(self):
        """
        Recalculates all tracking sets based *only* on the current grid state.
        This is a fallback/synchronization method. Ideally, modifications
        should update sets directly, making this less necessary.
        It does NOT update self.agents or self.bouncing_obstacle_objects.
        """
        if Config.DEBUG:
            print("Warning: update_all_sets called. This might indicate inconsistent state updates elsewhere.")
        self.empty_cells = set()
        self.food_cells = set()
        self.obstacle_cells = set()
        self.moveable_obstacle_cells = set()
        self.grabbable_obstacle_cells = set()
        # Note: Doesn't update bouncing obstacles objects or agent positions

        for r in range(self.size):
            for c in range(self.size):
                cell = (r, c)
                cell_type = CellType(self.grid[r, c])
                if cell_type == CellType.EMPTY:
                    self.empty_cells.add(cell)
                elif cell_type == CellType.FOOD:
                    self.food_cells.add(cell)
                elif cell_type == CellType.OBSTACLE:
                    self.obstacle_cells.add(cell)
                elif cell_type == CellType.MOVEABLE_OBSTACLE:
                    self.moveable_obstacle_cells.add(cell)
                elif cell_type == CellType.GRABBABLE_OBSTACLE:
                    self.grabbable_obstacle_cells.add(cell)
        self.food_remaining = len(self.food_cells)

class SnailPursuer:
    """Slow, immortal shortest-path pursuer occupying only empty cells.

    Food is deliberately impassable. The snail neither consumes, covers, nor
    displaces it, so food can create temporary path barriers together with the
    ordinary obstacle field.
    """

    def __init__(self, x: int, y: int, grid_size: int, move_interval: int = 10):
        if int(move_interval) < 1:
            raise ValueError("snail move_interval must be positive")
        self.x = int(x)
        self.y = int(y)
        self.grid_size = int(grid_size)
        self.move_interval = int(move_interval)
        self.moves_made = 0
        self.kills = 0

    @property
    def position(self) -> tuple[int, int]:
        return self.x, self.y

    @staticmethod
    def _passable(
        env: GridEnvironment,
        cell: tuple[int, int],
        goal: tuple[int, int],
    ) -> bool:
        r, c = cell
        if not env.is_within_bounds(r, c):
            return False
        if cell == goal:
            return CellType(env.grid[r, c]) == CellType.AGENT
        # This intentionally excludes FOOD as well as every obstacle type.
        return CellType(env.grid[r, c]) == CellType.EMPTY

    @classmethod
    def _distances_from_agent(
        cls,
        env: GridEnvironment,
        origin: tuple[int, int],
        max_distance: int,
    ) -> dict[tuple[int, int], int]:
        distances = {origin: 0}
        frontier = deque([origin])
        while frontier:
            current = frontier.popleft()
            if distances[current] >= max_distance:
                continue
            for dr, dc in CARDINAL_DELTAS:
                candidate = current[0] + dr, current[1] + dc
                if candidate in distances:
                    continue
                if not env.is_within_bounds(*candidate):
                    continue
                if CellType(env.grid[candidate]) != CellType.EMPTY:
                    continue
                distances[candidate] = distances[current] + 1
                frontier.append(candidate)
        return distances

    @classmethod
    def spawn(
        cls,
        env: GridEnvironment,
        agent: "Agent",
        *,
        min_distance: int,
        max_distance: int,
        move_interval: int,
        others: Optional[list["Agent"]] = None,
    ) -> Optional["SnailPursuer"]:
        if not 1 <= int(min_distance) <= int(max_distance):
            raise ValueError("snail spawn distances must satisfy 1 <= min <= max")
        distances = cls._distances_from_agent(
            env, (agent.x, agent.y), int(max_distance)
        )
        in_band = sorted(
            cell
            for cell, distance in distances.items()
            if int(min_distance) <= distance <= int(max_distance)
            and CellType(env.grid[cell]) == CellType.EMPTY
        )
        # Prefer a cell that is also outside every other resident's minimum
        # distance, so a pursuer does not materialize on someone it does not
        # hunt. On a small board with residents spawned close together no such
        # cell may exist, and the band around the hunted agent is the contract
        # that matters, so this is a preference rather than a requirement.
        too_close: set[tuple[int, int]] = set()
        for other in (others or []):
            too_close.update(
                cls._distances_from_agent(
                    env, (other.x, other.y), int(min_distance) - 1
                )
            )
        candidates = [cell for cell in in_band if cell not in too_close] or in_band
        if not candidates:
            return None
        x, y = random.choice(candidates)
        snail = cls(x, y, env.size, move_interval=move_interval)
        env._update_cell(x, y, CellType.PURSUER)
        return snail

    def plan_path(
        self,
        env: GridEnvironment,
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        start = self.position
        frontier = deque([start])
        parent: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
        while frontier:
            current = frontier.popleft()
            if current == goal:
                break
            for dr, dc in CARDINAL_DELTAS:
                candidate = current[0] + dr, current[1] + dc
                if candidate in parent or not self._passable(env, candidate, goal):
                    continue
                parent[candidate] = current
                frontier.append(candidate)
        if goal not in parent:
            return []
        path = []
        node: Optional[tuple[int, int]] = goal
        while node is not None and node != start:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    def update(self, env: GridEnvironment, agent: "Agent", tick: int) -> None:
        if tick <= 0 or tick % self.move_interval:
            return
        path = self.plan_path(env, (agent.x, agent.y))
        if not path:
            return
        target = path[0]
        self.moves_made += 1
        if target == (agent.x, agent.y):
            self.kills += 1
            env.queue_snail_hit(agent)
            return
        if CellType(env.grid[target]) != CellType.EMPTY:
            return
        env._update_cell(self.x, self.y, CellType.EMPTY)
        self.x, self.y = target
        env._update_cell(self.x, self.y, CellType.PURSUER)


class BouncingObstacle:
    """Mobile obstacle with stable identity, varied movement rules, and collision tracking."""
    # Class attribute for direction deltas
    ALL_DIRECTIONS_DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

    # Stable identity counter (picklable)
    _NEXT_ID = 0

    def __init__(self, initial_x, initial_y, initial_direction, grid_size):
        """
        Initializes a bouncing obstacle.

        Args:
            initial_x (int): Initial row position.
            initial_y (int): Initial column position.
            initial_direction (str): Initial direction ("up", "down", "left", "right").
            grid_size (int): The size of the grid the obstacle lives in.
        """
        # ---- stable identity for hashing/equality ----
        self._id = BouncingObstacle._NEXT_ID
        BouncingObstacle._NEXT_ID += 1

        # ---- mutable state ----
        # NOTE: x is the row index, y is the column index (matches grid[x, y] indexing).
        self.x = initial_x
        self.y = initial_y
        self.direction = initial_direction
        self.grid_size = grid_size

        self.type = random.choice(["rotating", "inverting", "counter_rotating", "wandering"])
        self.tick, self.grid_value = self.determine_type_properties()

        self.dead = False  # retained for checkpoint compatibility; runtime stalls instead of dying
        self.stalled_ticks = 0
        self.max_attempts = 4

    def __hash__(self):
        # stable for life of the object; independent of x/y/direction/type
        return hash(self._id)

    def __eq__(self, other):
        # identity equality: two different objects are never "equal"
        return isinstance(other, BouncingObstacle) and self._id == other._id

    def determine_type_properties(self):
        """Determines tick rate and grid value based on the obstacle type."""
        if self.type == "inverting":
            random_ticks = 1
            return random_ticks, CellType.BOUNCING_OBSTACLE
        if self.type in ["rotating", "counter_rotating"]:
            random_ticks = 1
            grid_value = CellType.LOOPING_OBSTACLE if self.type == "rotating" else CellType.COUNTER_LOOPING_OBSTACLE
            return random_ticks, grid_value
        if self.type == "wandering":
            random_ticks = 1
            return random_ticks, CellType.WANDERING_OBSTACLE
        return 1, CellType.BOUNCING_OBSTACLE

    OPPOSITE_DIRECTIONS = {"up": "down", "down": "up", "left": "right", "right": "left"}

    def invert_direction(self):
        """ Inverts the obstacle's direction. """
        self.direction = BouncingObstacle.OPPOSITE_DIRECTIONS.get(self.direction, self.direction)

    def rotate_direction(self, clockwise=True):
        """ Rotates the obstacle's direction (clockwise or counter-clockwise). """
        order = ["up", "right", "down", "left"]
        if not clockwise:
            order.reverse()
        try:
            current_index = order.index(self.direction)
            self.direction = order[(current_index + 1) % len(order)]
        except ValueError:
            print(f"Warning: Invalid direction '{self.direction}' encountered during rotation.")

    def handle_collision_behavior(self):
        """
        Handles behavior upon collision for the rotating / counter_rotating
        types. (wandering and inverting have their own dedicated handlers.)
        """
        if self.type == "rotating":
            self.rotate_direction(clockwise=True)
        elif self.type == "counter_rotating":
            self.rotate_direction(clockwise=False)

    def _handle_collision_behavior_wandering(self, env) -> bool:
        """
        Handles wandering collision by choosing a valid adjacent direction.
        A trapped obstacle stalls in place and retries on a later tick.
        """
        valid_directions = []
        for potential_dir, (dr, dc) in BouncingObstacle.ALL_DIRECTIONS_DELTAS.items():
            next_x, next_y = self.x + dr, self.y + dc
            if 0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size:
                if env.grid[next_x, next_y] == CellType.EMPTY.value:
                    valid_directions.append(potential_dir)

        if valid_directions:
            self.direction = random.choice(valid_directions)
            return False  # not dead
        else:
            self.stalled_ticks += 1
            return True   # stalled this tick, but remains present

    def _handle_collision_behavior_inverting(self, env) -> bool:
        """
        Handles collision for the inverting type. Its signature move is to bounce
        straight back the way it came, so that's tried first. Only if BOTH
        directions along its axis are blocked (e.g. pinned between two neighbors)
        does it fall back to a perpendicular escape, rather than dying just for
        being wedged between two other obstacles while open cells sit beside it.
        """
        self.invert_direction()
        dr, dc = BouncingObstacle.ALL_DIRECTIONS_DELTAS[self.direction]
        next_x, next_y = self.x + dr, self.y + dc
        if 0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size:
            if env.grid[next_x, next_y] == CellType.EMPTY.value:
                return False  # opposite direction is clear; retry loop will take it

        axis = {self.direction, BouncingObstacle.OPPOSITE_DIRECTIONS[self.direction]}
        perpendicular_options = []
        for potential_dir, (pdr, pdc) in BouncingObstacle.ALL_DIRECTIONS_DELTAS.items():
            if potential_dir in axis:
                continue
            next_x, next_y = self.x + pdr, self.y + pdc
            if 0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size:
                if env.grid[next_x, next_y] == CellType.EMPTY.value:
                    perpendicular_options.append(potential_dir)

        if perpendicular_options:
            self.direction = random.choice(perpendicular_options)
            return False  # not dead

        self.stalled_ticks += 1
        return True  # boxed in for this tick; do not remove the obstacle

    def update(self, env, tick: int):
        """Attempt movement with type-specific reflections; stay put if trapped."""
        if tick % self.tick != 0:
            return
        collision_recorded = False
        tried: set[tuple[int, int, str]] = set()
        for _ in range(self.max_attempts):
            pending_x, pending_y = self.update_position_based_on_direction()
            state = (pending_x, pending_y, self.direction)
            if state in tried:
                break
            tried.add(state)
            in_bounds = env.is_within_bounds(pending_x, pending_y)
            target = CellType(env.grid[pending_x, pending_y]) if in_bounds else None
            if in_bounds and target == CellType.EMPTY:
                old = (self.x, self.y)
                env._update_cell(old[0], old[1], CellType.EMPTY)
                self.x, self.y = pending_x, pending_y
                env._update_cell(self.x, self.y, self.grid_value)
                self.stalled_ticks = 0
                return
            if in_bounds and target == CellType.AGENT and not collision_recorded:
                env.queue_bouncer_hit(pending_x, pending_y, self._id)
                collision_recorded = True

            if self.type == "wandering":
                if self._handle_collision_behavior_wandering(env):
                    break
            elif self.type == "inverting":
                if self._handle_collision_behavior_inverting(env):
                    break
            else:
                self.handle_collision_behavior()

        self.stalled_ticks += 1
        # Deliberately remain on the grid. A temporary trap is an obstacle state,
        # not an object-lifetime event.

    def update_position_based_on_direction(self):
        """ Calculates the potential next position based on the current direction. """
        dr, dc = BouncingObstacle.ALL_DIRECTIONS_DELTAS.get(self.direction, (0, 0))
        return self.x + dr, self.y + dc

