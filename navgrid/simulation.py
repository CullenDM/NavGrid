"""Environment simulation runner and multi-agent coordinator for NavGrid."""

from __future__ import annotations

import copy
import gc
import math
import os
import queue
import random
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import optim

from .agent import Agent
from .config import Config
from .constants import (
    Action,
    CellType,
    Direction,
    EpisodeVariant,
    FoodMovementMode,
    ObstacleSetupChoice,
)
from .environment import (
    CurriculumScenario,
    GridEnvironment,
    uses_competitive_food_share_completion,
)
from .model import PPOAgent
from .ppo import PPO
from .visualizer import (
    EnvRenderSnapshot,
    GridVisualizer,
    PanelData,
    RenderSnapshot,
    render_mainloop,
)

class EnvironmentSimulation:
    """Orchestrates environments, agents, training, and the render loop split."""
    def __init__(self):
        self._validate_episode_batch_profile()
        self.envs = self.initialize_environments()
        self.visualizer = self.initialize_visualizer(self.envs)
        self.model = self.initialize_model()
        self.ppo = PPO(self.model, self.optimizer, Config.CLIP_PARAM, Config.PPO_EPOCHS, Config.TARGET_KL, Config.GAMMA, Config.TAU)
        self._init_curriculum_state()

        self.agents = self.initialize_agents(self.model, self.envs)
        self._last_snapshot_sig = None
        self._in_update = False
        self._training_overlay_frames = 0
        self.snapshot_q = None
        self.current_episode = 1
        self.episode_start_time = time.monotonic()
        self.last_losses = (0.0, 0.0, 0.0)
        self.last_compute_time = 0.0
        self.update_count = 0
        self.rollout_batch_episodes = 0
        self.last_rollout_batch_samples = 0
        self.last_rollout_update_episode = 0
        self.best_eval_reward = -float('inf')
        self.last_eval_update = 0
        self.eval_regression_streak = 0
        self.regression_halted = False
        self._legacy_resume_boundary_pending = False
        self._rss_samples = deque(
            maxlen=max(2, int(getattr(Config, "MEMORY_GROWTH_WINDOW", 12)))
        )
        if Config.LOAD_MODEL:
            self.load_checkpoint(Config.MODEL_PATH)

    @staticmethod
    def _validate_episode_batch_profile() -> None:
        """Reject settings that violate this standalone trainer's contract."""
        if int(Config.NUM_ENVS) != 1:
            raise ValueError(
                "episode-batch trainer requires exactly one environment"
            )
        if int(Config.NUM_AGENTS) < 1:
            raise ValueError("NUM_AGENTS must be at least one")
        if int(getattr(Config, "MAX_EPISODE_STEPS", 0)) < 0:
            raise ValueError("MAX_EPISODE_STEPS must be zero (no cap) or positive")
        if int(getattr(Config, "PPO_EPISODES_PER_UPDATE", 0)) <= 0:
            raise ValueError("PPO_EPISODES_PER_UPDATE must be positive")

    @staticmethod
    def _current_rss_mib() -> float:
        """Read current resident memory without adding a psutil dependency."""
        try:
            with open("/proc/self/statm", "r", encoding="ascii") as handle:
                resident_pages = int(handle.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 ** 2)
        except (OSError, ValueError, IndexError):
            return float("nan")

    def _report_memory(self, rollout_samples: int) -> None:
        interval = max(
            0, int(getattr(Config, "MEMORY_REPORT_INTERVAL_UPDATES", 0))
        )
        if not interval or self.update_count % interval:
            return
        rss_mib = self._current_rss_mib()
        self._rss_samples.append(rss_mib)
        print(
            f"[MEMORY] update={self.update_count} rss={rss_mib:.1f} MiB "
            f"rollout={rollout_samples} minibatch={Config.MINIBATCH_SIZE}",
            flush=True,
        )
        if len(self._rss_samples) == self._rss_samples.maxlen:
            growth = self._rss_samples[-1] - self._rss_samples[0]
            warning_threshold = float(
                getattr(Config, "MEMORY_GROWTH_WARN_MIB", 512.0)
            )
            if growth > warning_threshold and all(
                newer >= older
                for older, newer in zip(self._rss_samples, list(self._rss_samples)[1:])
            ):
                print(
                    f"[MEMORY WARNING] RSS grew monotonically by {growth:.1f} MiB "
                    f"across {len(self._rss_samples)} reports",
                    flush=True,
                )

    def initialize_environments(self):
        return [GridEnvironment(idx) for idx in range(Config.NUM_ENVS)]

    def initialize_visualizer(self, envs):
        return GridVisualizer(len(envs), envs[0].size, Config.CELL_SIZE) if Config.VISUALIZE else None

    def initialize_model(self):
        if getattr(Config, "RANDOM_ACTION_POLICY", False) and Config.USE_PPO_UPDATE:
            print("[RANDOM_ACTION_POLICY] forcing USE_PPO_UPDATE=False", flush=True)
            Config.USE_PPO_UPDATE = False
        model = PPOAgent()
        calculated_trainable_params = model.calculate_trainable_parameters()
        print(f"Calculated trainable parameters: {calculated_trainable_params}")
        self.optimizer = self.initialize_optimizer(model)
        if (os.environ.get("NAVGRID_COMPILE", "0") == "1"
                and not getattr(Config, "RANDOM_ACTION_POLICY", False)):
            import torch._dynamo
            # The default ceiling of 8 crashes this worker; PPO has several
            # execution signatures (rollout, update, diagnostics).
            for attribute in ("recompile_limit", "cache_size_limit"):
                if hasattr(torch._dynamo.config, attribute):
                    setattr(torch._dynamo.config, attribute, 32)
            # Deliberately NOT dynamic=False -- the PPO tail minibatch is ragged,
            # so static shapes would recompile instead of specialising once.
            model.compile(fullgraph=True)
            print("[compile] inductor fullgraph=True, recompile_limit=32", flush=True)
        return model

    def initialize_optimizer(self, model):
        parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
        optimizer = optim.Adam(
            [{
                "params": parameters,
                "lr": Config.LEARNING_RATE,
                "lr_scale": 1.0,
                "group_name": "model",
            }],
            lr=Config.LEARNING_RATE,
            weight_decay=0.0,
        )
        expected = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        assigned = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        if len(assigned) != len(set(assigned)) or set(assigned) != expected:
            raise RuntimeError(
                "V32 nav optimizer must contain every trainable parameter "
                "exactly once"
            )
        print(
            f"[{Config.MODEL_NAME} OPTIMIZER] "
            f"lr={Config.LEARNING_RATE:.3e} | "
            f"params={sum(p.numel() for p in parameters):,}",
            flush=True,
        )
        return optimizer

    def initialize_agents(self, model, envs, visualizer=None):
        agents = []
        for env_idx, env in enumerate(envs):
            for agent_idx in range(Config.NUM_AGENTS):
                agent = Agent(model, env, env_idx, agent_idx, visualizer)
                agents.append(agent)
        return agents

    @staticmethod
    def _draw_episode_variant(*, evaluation: bool) -> EpisodeVariant:
        """Draw one exclusive episode condition from the training RNG stream."""
        if evaluation:
            return EpisodeVariant(
                str(getattr(Config, "EVALUATION_ARENA_VARIANT", "standard"))
            )
        if not bool(getattr(Config, "ARENA_VARIANTS_ENABLED", False)):
            return EpisodeVariant.STANDARD

        snail_probability = (
            float(getattr(Config, "IMMORTAL_SNAIL_EPISODE_PROBABILITY", 0.0))
            if bool(getattr(Config, "USE_IMMORTAL_SNAIL", False))
            else 0.0
        )
        dual_probability = (
            float(getattr(Config, "AGENT_VERSUS_AGENT_EPISODE_PROBABILITY", 0.0))
            if bool(getattr(Config, "USE_AGENT_PREDATION", False))
            else 0.0
        )
        if not 0.0 <= snail_probability <= 1.0:
            raise ValueError("IMMORTAL_SNAIL_EPISODE_PROBABILITY must be in [0, 1]")
        if not 0.0 <= dual_probability <= 1.0:
            raise ValueError("AGENT_VERSUS_AGENT_EPISODE_PROBABILITY must be in [0, 1]")
        if snail_probability + dual_probability > 1.0:
            raise ValueError("exclusive arena episode probabilities must sum to <= 1")

        draw = random.random()
        if draw < snail_probability:
            return EpisodeVariant.IMMORTAL_SNAIL
        if draw < snail_probability + dual_probability:
            return EpisodeVariant.DUAL_AGENT
        return EpisodeVariant.STANDARD

    def reset_agents(self, agents, *, evaluation: bool = False):
        by_env = {}
        for agent in agents:
            by_env.setdefault(id(agent.env), []).append(agent)
        for residents in by_env.values():
            residents.sort(key=lambda agent: int(agent.agent_idx))
            env = residents[0].env
            variant = self._draw_episode_variant(evaluation=evaluation)
            if variant is EpisodeVariant.DUAL_AGENT:
                enough_residents = len(residents) >= 2
                enough_spawn_cells = len(env.get_spawnable_cells()) >= 2
                if not enough_residents or not enough_spawn_cells:
                    print(
                        f"[EPISODE VARIANT FALLBACK] env{env.idx}=dual_agent -> "
                        "standard (insufficient residents or spawn cells)",
                        flush=True,
                    )
                    variant = EpisodeVariant.STANDARD
            participant_count = 2 if variant is EpisodeVariant.DUAL_AGENT else 1
            require_snail = bool(
                getattr(Config, "REQUIRE_SNAIL_ON_SNAIL_EPISODE", True)
            ) and variant is EpisodeVariant.IMMORTAL_SNAIL
            attempts = (
                max(
                    1,
                    int(
                        getattr(
                            Config,
                            "IMMORTAL_SNAIL_GENERATION_ATTEMPTS",
                            20,
                        )
                    ),
                )
                if require_snail
                else 1
            )
            for attempt in range(1, attempts + 1):
                if attempt > 1:
                    # Keep the already-drawn curriculum and episode variant;
                    # only regenerate board geometry and placements.
                    env.reset()
                env.configure_episode_variant(variant)
                for index, agent in enumerate(residents):
                    agent.reset(participating=index < participant_count)
                env.new_grid_generated = False
                env.configure_snail_for_episode()
                snail_ready = (
                    variant is not EpisodeVariant.IMMORTAL_SNAIL
                    or len(env.snail_pursuers) == participant_count
                )
                if snail_ready or not require_snail:
                    break
                if attempt == attempts:
                    raise RuntimeError(
                        f"Env {env.idx} could not place its required snail "
                        f"after {attempts} generated boards"
                    )
                print(
                    f"[SNAIL REQUIRED] env{env.idx} regenerating board after "
                    f"spawn-band failure ({attempt}/{attempts})",
                    flush=True,
                )
            mode = "eval" if evaluation else "train"
            print(
                f"[EPISODE VARIANT {mode}] env{env.idx}={variant.value} "
                f"participants={participant_count}",
                flush=True,
            )

    @staticmethod
    def _context_choices() -> tuple[tuple[int, ...], tuple[float, ...]]:
        choices = tuple(
            int(value)
            for value in getattr(
                Config, "SEQUENCE_LENGTH_CHOICES", (Config.SEQUENCE_LENGTH,)
            )
        )
        weights = tuple(
            float(value)
            for value in getattr(
                Config, "SEQUENCE_LENGTH_WEIGHTS", (1.0,) * len(choices)
            )
        )
        if not choices:
            raise ValueError("SEQUENCE_LENGTH_CHOICES must not be empty")
        if len(weights) != len(choices):
            raise ValueError(
                "SEQUENCE_LENGTH_WEIGHTS must match SEQUENCE_LENGTH_CHOICES"
            )
        if any(value < 1 or value > int(Config.SEQUENCE_LENGTH) for value in choices):
            raise ValueError(
                "Every context choice must be between 1 and SEQUENCE_LENGTH="
                f"{Config.SEQUENCE_LENGTH}"
            )
        if any(weight <= 0.0 for weight in weights):
            raise ValueError("SEQUENCE_LENGTH_WEIGHTS must all be positive")

        food_minimum = float(Config.FOOD_CHAIN_MIN_MULTIPLIER)
        food_maximum = float(Config.FOOD_CHAIN_MAX_MULTIPLIER)
        if not 0.0 < food_minimum <= food_maximum:
            raise ValueError(
                "Food-chain bounds must satisfy "
                "0 < FOOD_CHAIN_MIN_MULTIPLIER <= FOOD_CHAIN_MAX_MULTIPLIER"
            )
        return choices, weights

    def assign_context_lengths(self, envs, *, evaluation: bool) -> None:
        """Choose one usable temporal horizon independently for each environment."""
        choices, weights = self._context_choices()
        evaluation_length = int(
            getattr(Config, "EVAL_SEQUENCE_LENGTH", max(choices))
        )
        if evaluation_length not in choices:
            raise ValueError(
                f"EVAL_SEQUENCE_LENGTH={evaluation_length} is not one of {choices}"
            )
        assignments = []
        for env in envs:
            length = (
                evaluation_length
                if evaluation
                else int(random.choices(choices, weights=weights, k=1)[0])
            )
            env.sequence_length = length
            assignments.append(f"env{env.idx}={length}")
        mode = "eval" if evaluation else "train"
        print(f"[CONTEXT {mode}] " + " ".join(assignments), flush=True)

    def run(self):
        stop_event = threading.Event()
        snapshot_q = queue.Queue(maxsize=1) if Config.VISUALIZE else None

        if Config.VISUALIZE and self.visualizer is not None:
            worker_errors = []

            def run_training_worker():
                try:
                    self.run_training(stop_event, snapshot_q)
                except BaseException as error:
                    worker_errors.append(error)
                    stop_event.set()
                    raise

            worker = threading.Thread(
                target=run_training_worker,
                daemon=True,
            )
            worker.start()
            try:
                render_mainloop(self.visualizer, snapshot_q, stop_event)
            finally:
                stop_event.set()
                worker.join(timeout=5)
                self.visualizer.close()
            if worker_errors:
                raise RuntimeError(
                    "navigation training worker failed"
                ) from worker_errors[0]
        else:
            try:
                self.run_training(stop_event, snapshot_q)
            finally:
                stop_event.set()

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---------------------------------------------------------------------------
    # Curriculum scenario pool with optional frontier gating.
    # ---------------------------------------------------------------------------
    CURRICULUM_SCENARIOS: tuple[CurriculumScenario, ...] = (
        CurriculumScenario("static food / empty", 0, False, "random", ObstacleSetupChoice.EMPTY_NO_OBSTACLES, 0, 0.50),
        CurriculumScenario("static food / grabbable", 0, False, "random", ObstacleSetupChoice.ONLY_GRABBABLE_LARGE, 0, 0.75),
        CurriculumScenario("static food / early bouncers", 0, False, "random", ObstacleSetupChoice.MIXED_SMALL_BOUNCE_LOW, 0, 1.25),
        CurriculumScenario("moving food / empty", 0, True, "random", ObstacleSetupChoice.EMPTY_NO_OBSTACLES, 4, 1.25),
        CurriculumScenario("static food / moveable", 1, False, "random", ObstacleSetupChoice.ONLY_MOVEABLE_LARGE, 0),
        CurriculumScenario("static food / moveable+grab", 1, False, "random", ObstacleSetupChoice.MOVEABLE_GRABBABLE_HALF, 0),
        CurriculumScenario("boids food / empty", 1, True, "boids", ObstacleSetupChoice.EMPTY_NO_OBSTACLES, 4),
        CurriculumScenario("moving food / early bouncers", 1, True, "random", ObstacleSetupChoice.MIXED_SMALL_BOUNCE_LOW, 4),
        CurriculumScenario("static food / mixed no-bounce", 2, False, "random", ObstacleSetupChoice.MIXED_SMALL_NO_BOUNCE, 0),
        CurriculumScenario("static food / mixed+bounce", 2, False, "random", ObstacleSetupChoice.MIXED_SMALL_BOUNCE_MED, 0),
        CurriculumScenario("moving food / moveable+grab", 2, True, "random", ObstacleSetupChoice.MOVEABLE_GRABBABLE_HALF, 4),
        CurriculumScenario("boids food / moveable+grab", 2, True, "boids", ObstacleSetupChoice.MOVEABLE_GRABBABLE_HALF, 4),
        CurriculumScenario("moving food / mixed", 3, True, "random", ObstacleSetupChoice.MIXED_ALL_TYPES, 4),
        CurriculumScenario("boids food / mixed", 3, True, "boids", ObstacleSetupChoice.MIXED_ALL_TYPES, 4),
        CurriculumScenario("static food / moveable-half", 1, False, "random", ObstacleSetupChoice.STATIC_MOVEABLE_HALF, 0),
    )
    ENV_SCALE_STEPS = (
        (15, 12),
        (21, 9),
        (29, 7),
        (37, 5),
    )

    def _init_curriculum_state(self) -> None:
        window = max(1, int(Config.CURRICULUM_WINDOW))
        self.curriculum_history = [deque(maxlen=window) for _ in self.CURRICULUM_SCENARIOS]
        self.config_mastered = [False] * len(self.CURRICULUM_SCENARIOS)
        self.curriculum_level = min(s.difficulty for s in self.CURRICULUM_SCENARIOS)
        self.curriculum_idx = next(
            i for i, scenario in enumerate(self.CURRICULUM_SCENARIOS)
            if scenario.difficulty == self.curriculum_level
        )
        self.curriculum_episode_counter = 0
        self.env_scale_idx = -1
        self._pending_scale: Optional[tuple[int, int]] = None
        self._scale_complete = False

    def _scenario_rate(self, idx: int) -> float:
        history = self.curriculum_history[idx]
        return float(sum(history)) / float(len(history)) if history else 0.0

    def _scenario_mastered(self, idx: int) -> bool:
        history = self.curriculum_history[idx]
        return (
            len(history) >= int(Config.CURRICULUM_MIN_TRIALS)
            and self._scenario_rate(idx) >= float(Config.CURRICULUM_MASTERY_RATE)
        )

    def _refresh_curriculum_mastery(self) -> None:
        for idx in range(len(self.CURRICULUM_SCENARIOS)):
            self.config_mastered[idx] = self._scenario_mastered(idx)

    def assign_curriculum_scenarios(
        self,
        envs: Sequence[GridEnvironment],
        *,
        evaluation: bool,
    ) -> None:
        """Give each environment a unique, difficulty-balanced scenario."""
        if (
            not bool(Config.CURRICULUM_ENABLED)
            or not bool(getattr(Config, "CURRICULUM_PER_ENV_MIX", False))
        ):
            return

        difficulties = sorted({item.difficulty for item in self.CURRICULUM_SCENARIOS})
        difficulty_sequence = [
            difficulties[idx % len(difficulties)]
            for idx in range(len(envs))
        ]
        if not evaluation:
            random.shuffle(difficulty_sequence)

        used: set[int] = set()
        assignments: list[tuple[GridEnvironment, int]] = []
        for env, difficulty in zip(envs, difficulty_sequence):
            candidates = [
                idx
                for idx, scenario in enumerate(self.CURRICULUM_SCENARIOS)
                if scenario.difficulty == difficulty and idx not in used
            ]
            if not candidates:
                raise ValueError(
                    "CURRICULUM_PER_ENV_MIX requires enough unique scenarios "
                    f"at L{difficulty} for {len(envs)} environments"
                )
            scenario_idx = candidates[0] if evaluation else self._weighted_choice(candidates)
            used.add(scenario_idx)
            env.configure_curriculum_scenario(
                scenario_idx,
                self.CURRICULUM_SCENARIOS[scenario_idx],
            )
            assignments.append((env, scenario_idx))

        mode = "eval" if evaluation else "train"
        summary = " | ".join(
            f"env{env.idx}=L{self.CURRICULUM_SCENARIOS[idx].difficulty} "
            f"{self.CURRICULUM_SCENARIOS[idx].label}"
            for env, idx in assignments
        )
        print(f"[CURRICULUM MIX {mode}] {summary}", flush=True)

    def _apply_curriculum_config(self, idx: int) -> None:
        scenario = self.CURRICULUM_SCENARIOS[idx]
        Config.FOOD_TICK = scenario.food_tick
        Config.FOOD_MOVEMENT_MODE = scenario.food_mode
        Config.OBSTACLE_CHOICE = int(scenario.obstacle_choice)
        Config.FOOD_TICK_SPEED = scenario.food_tick_speed if scenario.food_tick else 0
        Config.MAX_FOOD_TICK_SPEED = scenario.food_tick_speed if scenario.food_tick else 0
        history = self.curriculum_history[idx]
        curriculum_scope = (
            "pool=all"
            if bool(Config.CURRICULUM_ALL_SCENARIOS)
            else f"frontier=L{self.curriculum_level}"
        )
        print(
            f"[CURRICULUM] #{idx:02d} L{scenario.difficulty} {scenario.label} | "
            f"rate={self._scenario_rate(idx):.2f} n={len(history)} | "
            f"{curriculum_scope}",
            flush=True,
        )

    def _weighted_choice(self, indices: Sequence[int]) -> int:
        weights = []
        min_trials = max(1, int(Config.CURRICULUM_MIN_TRIALS))
        for idx in indices:
            scenario = self.CURRICULUM_SCENARIOS[idx]
            history = self.curriculum_history[idx]
            uncertainty = max(0.0, 1.0 - len(history) / min_trials)
            difficulty_need = max(0.05, 1.0 - self._scenario_rate(idx))
            weights.append(float(scenario.weight) * (0.25 + uncertainty + difficulty_need))
        return random.choices(list(indices), weights=weights, k=1)[0]

    def _choose_next_curriculum_idx(self) -> int:
        if bool(Config.CURRICULUM_ALL_SCENARIOS):
            indices = list(range(len(self.CURRICULUM_SCENARIOS)))
            unseen = [idx for idx in indices if not self.curriculum_history[idx]]
            return self._weighted_choice(unseen if unseen else indices)

        all_mastered = all(self.config_mastered)
        max_level = max(s.difficulty for s in self.CURRICULUM_SCENARIOS)

        # Guarantee one exposure to every newly unlocked regime before weighted
        # replay can repeatedly select an easier sibling.
        unseen_frontier = [
            idx for idx, scenario in enumerate(self.CURRICULUM_SCENARIOS)
            if scenario.difficulty == self.curriculum_level
            and not self.curriculum_history[idx]
        ]
        if unseen_frontier:
            return self._weighted_choice(unseen_frontier)
        
        indices = list(range(len(self.CURRICULUM_SCENARIOS)))
        weights = []
        
        for idx in indices:
            scenario = self.CURRICULUM_SCENARIOS[idx]
            history = self.curriculum_history[idx]
            
            # Base weight from scenario config (usually 1.0)
            w = float(scenario.weight)
            
            if all_mastered and self.curriculum_level == max_level:
                # End of run: even mix of everything!
                weights.append(w)
                continue
                
            # Not fully mastered yet
            if scenario.difficulty < self.curriculum_level:
                # REPLAY: ensure no collapse of old skills
                retention = float(getattr(Config, "CURRICULUM_RETENTION_RATE", 0.8))
                rate = self._scenario_rate(idx)
                uncertainty = max(0.0, 1.0 - len(history) / max(1, int(Config.CURRICULUM_MIN_TRIALS)))
                difficulty_need = max(0.0, retention - rate) # Boost if rate is below retention
                
                # Normal weight is w. We boost it slightly if it's forgotten.
                w *= (1.0 + uncertainty + difficulty_need)
            elif scenario.difficulty == self.curriculum_level:
                # FRONTIER: heavy focus to push escalation
                rate = self._scenario_rate(idx)
                uncertainty = max(0.0, 1.0 - len(history) / max(1, int(Config.CURRICULUM_MIN_TRIALS)))
                difficulty_need = max(0.05, 1.0 - rate)
                
                # 5x base multiplier to prioritize frontier progression
                w *= 5.0 * (0.5 + uncertainty + difficulty_need)
            else:
                # REACH: Symmetry breaker
                # Drop a future difficulty randomly (5% base relative weight)
                w *= 0.05
            
            weights.append(w)
            
        return random.choices(indices, weights=weights, k=1)[0]

    def _record_curriculum_result(self, success_fraction: float) -> None:
        idx = self.curriculum_idx
        success = success_fraction >= float(Config.CURRICULUM_ENV_SUCCESS_FRACTION)
        self.curriculum_history[idx].append(bool(success))
        self.curriculum_episode_counter += 1
        self._refresh_curriculum_mastery()
        scenario = self.CURRICULUM_SCENARIOS[idx]
        print(
            f"[CURRICULUM RESULT] #{idx:02d} {scenario.label}: "
            f"env-clear={success_fraction:.2f} rolling={self._scenario_rate(idx):.2f} "
            f"n={len(self.curriculum_history[idx])}",
            flush=True,
        )

        if bool(Config.CURRICULUM_ALL_SCENARIOS):
            if all(self.config_mastered):
                self._schedule_scale_up()
            return

        frontier = [
            i for i, item in enumerate(self.CURRICULUM_SCENARIOS)
            if item.difficulty == self.curriculum_level
        ]
        if frontier and all(self.config_mastered[i] for i in frontier):
            max_level = max(item.difficulty for item in self.CURRICULUM_SCENARIOS)
            if self.curriculum_level < max_level:
                self.curriculum_level += 1
                print(f"[CURRICULUM UNLOCK] frontier -> L{self.curriculum_level}", flush=True)
            elif all(self.config_mastered):
                self._schedule_scale_up()

    def _record_per_env_curriculum_results(
        self,
        envs: Sequence[GridEnvironment],
    ) -> None:
        results = []
        for env in envs:
            idx = env.curriculum_scenario_idx
            if idx is None:
                raise RuntimeError(
                    f"Env {env.idx} has no assigned curriculum scenario"
                )
            success = bool(env.episode_cleared)
            self.curriculum_history[idx].append(success)
            scenario = self.CURRICULUM_SCENARIOS[idx]
            results.append(
                f"env{env.idx} #{idx:02d} L{scenario.difficulty}="
                f"{'clear' if success else 'fail'} "
                f"(rolling={self._scenario_rate(idx):.2f}, "
                f"n={len(self.curriculum_history[idx])})"
            )

        self.curriculum_episode_counter += len(envs)
        self._refresh_curriculum_mastery()
        print("[CURRICULUM MIX RESULT] " + " | ".join(results), flush=True)
        if all(self.config_mastered):
            self._schedule_scale_up()

    def _schedule_scale_up(self) -> None:
        if not bool(Config.CURRICULUM_SCALE_ENABLED) or self._scale_complete:
            return
        next_idx = self.env_scale_idx + 1
        if next_idx >= len(self.ENV_SCALE_STEPS):
            self._scale_complete = True
            print("[CURRICULUM] maximum environment scale retained", flush=True)
            return
        self.env_scale_idx = next_idx
        self._pending_scale = self.ENV_SCALE_STEPS[next_idx]

    def _apply_pending_scale_rebuild(self) -> None:
        if self._pending_scale is None:
            return
        new_size, new_cell = self._pending_scale
        self._pending_scale = None
        Config.ENVIRONMENT_SIZE = int(new_size)
        Config.CELL_SIZE = int(new_cell)
        # Episode boundary: discard no live state, keep model/optimizer/PPO intact.
        self.envs = self.initialize_environments()
        self.agents = self.initialize_agents(self.model, self.envs, self.visualizer)
        self._last_snapshot_sig = None
        self.curriculum_history = [
            deque(maxlen=max(1, int(Config.CURRICULUM_WINDOW)))
            for _ in self.CURRICULUM_SCENARIOS
        ]
        self.config_mastered = [False] * len(self.CURRICULUM_SCENARIOS)
        self.curriculum_level = min(s.difficulty for s in self.CURRICULUM_SCENARIOS)
        print(
            f"[CURRICULUM SCALE-UP] rebuilt environments at {new_size}x{new_size}, "
            f"cell={new_cell}px",
            flush=True,
        )

    def run_training(self, stop_event, snapshot_q=None):
        self.snapshot_q = snapshot_q
        episode = max(1, int(self.current_episode))
        target_render_fps = Config.FPS if Config.USE_FPS else 30
        per_env_mix = bool(getattr(Config, "CURRICULUM_PER_ENV_MIX", False))
        if self._legacy_resume_boundary_pending:
            # Older checkpoints were written before the curriculum draw and
            # periodic evaluation belonging to the completed episode boundary.
            self._finalize_episode_boundary(episode - 1, per_env_mix)
            self._legacy_resume_boundary_pending = False
            print(
                "[RESUME] completed legacy checkpoint's pending episode boundary",
                flush=True,
            )
        elif Config.CURRICULUM_ENABLED and not per_env_mix:
            self._apply_curriculum_config(self.curriculum_idx)

        if not Config.LOAD_MODEL:
            self.run_evaluation(0)

        try:
            while not stop_event.is_set():
                self.print_episode_header(episode)
                self.run_episode(stop_event, snapshot_q, target_render_fps)
                if stop_event.is_set():
                    break

                if Config.CURRICULUM_ENABLED:
                    if per_env_mix:
                        self._record_per_env_curriculum_results(self.envs)
                    else:
                        cleared = sum(1 for env in self.envs if env.episode_cleared)
                        success_fraction = cleared / max(1, len(self.envs))
                        self._record_curriculum_result(success_fraction)

                self._in_update = False
                self._training_overlay_frames = 0
                self.print_episode_summary()

                # A checkpoint represents the exact start of episode+1. Finish
                # every RNG-consuming boundary action before serializing it.
                self._finalize_episode_boundary(episode, per_env_mix)
                if self.regression_halted:
                    print(
                        "[EVAL SAFEGUARD] training frozen; the saved best "
                        "checkpoint remains the recovery point",
                        flush=True,
                    )
                    stop_event.set()
                    break
                if self.should_save_model(episode):
                    self.save_checkpoint(episode)

                episode += 1
                self.current_episode = episode
        except KeyboardInterrupt:
            print("\n[INFO] Caught KeyboardInterrupt. Shutting down worker...")
            stop_event.set()

    def _finalize_episode_boundary(
        self,
        completed_episode: int,
        per_env_mix: bool,
    ) -> None:
        """Advance all persistent state to the start of the next episode."""
        self._apply_pending_scale_rebuild()
        if Config.CURRICULUM_ENABLED and not per_env_mix:
            self.curriculum_idx = self._choose_next_curriculum_idx()
            self._apply_curriculum_config(self.curriculum_idx)

        current_updates = self.ppo.update_counter
        eval_freq = int(getattr(Config, "EVAL_FREQUENCY_UPDATES", 25))
        if current_updates - self.last_eval_update >= eval_freq:
            self.last_eval_update = current_updates
            self.run_evaluation(completed_episode)

    def run_evaluation(self, episode):
        if not getattr(self, "_multi_eval_active", False):
            seeds = tuple(
                int(seed) for seed in getattr(Config, "EVAL_SEEDS", (42,))
            )
            if not seeds:
                raise ValueError("EVAL_SEEDS must contain at least one seed")
            self._multi_eval_active = True
            rates = []
            try:
                for seed in seeds:
                    self._eval_seed = seed
                    rates.append(float(self.run_evaluation(episode)))
            finally:
                self._multi_eval_active = False
                if hasattr(self, "_eval_seed"):
                    del self._eval_seed

            eval_rate = sum(rates) / len(rates)
            print(
                f"[MULTI-SEED EVAL] seeds={seeds} "
                f"rates={[round(rate, 5) for rate in rates]} "
                f"mean={eval_rate:.5f}",
                flush=True,
            )
            previous_best = float(self.best_eval_reward)
            if eval_rate > previous_best:
                print(
                    f"New best multi-seed evaluation rate! "
                    f"{previous_best:.5f} -> {eval_rate:.5f}",
                    flush=True,
                )
                self.best_eval_reward = eval_rate
                self.eval_regression_streak = 0
                if not getattr(Config, "RANDOM_ACTION_POLICY", False):
                    best_path = f"models/ppo_model_{Config.MODEL_NAME}_best.pth"
                    self._atomic_torch_save(
                        self._checkpoint_payload(episode), best_path
                    )
                    print(f"Saved best model to {best_path}", flush=True)
            else:
                minimum_best = float(
                    getattr(Config, "EVAL_REGRESSION_MIN_BEST_RATE", 0.02)
                )
                fraction = float(
                    getattr(Config, "EVAL_REGRESSION_FRACTION", 0.25)
                )
                severe = (
                    math.isfinite(previous_best)
                    and previous_best >= minimum_best
                    and eval_rate < previous_best * fraction
                )
                self.eval_regression_streak = (
                    self.eval_regression_streak + 1 if severe else 0
                )
                patience = max(
                    1, int(getattr(Config, "EVAL_REGRESSION_PATIENCE", 3))
                )
                if self.eval_regression_streak >= patience:
                    self.regression_halted = True
                    print(
                        f"[EVAL SAFEGUARD] severe regression persisted for "
                        f"{self.eval_regression_streak} evaluations: "
                        f"mean={eval_rate:.5f}, best={previous_best:.5f}",
                        flush=True,
                    )
            return eval_rate

        eval_seed = int(getattr(self, "_eval_seed", 42))
        print(
            f"\n--- Running Evaluation at Episode {episode}, seed {eval_seed} ---"
        )
        
        # Save RNG states for true isolation
        rng_state_torch = torch.get_rng_state()
        rng_state_cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        rng_state_np = np.random.get_state()
        rng_state_py = random.getstate()
        
        # Fixed evaluation seed for deterministic rollouts
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(eval_seed)
        np.random.seed(eval_seed)
        random.seed(eval_seed)
        
        eval_envs = self.initialize_environments()
        eval_agents = self.initialize_agents(self.model, eval_envs)
        self.assign_curriculum_scenarios(eval_envs, evaluation=True)
        for env in eval_envs:
            env.reset()
        self.assign_context_lengths(eval_envs, evaluation=True)
        self.reset_agents(eval_agents, evaluation=True)
        for env in eval_envs:
            pack = env.snail_pursuers
            snail_state = (
                "active " + " ".join(
                    f"a{idx}:pos={snail.position}"
                    for idx, snail in sorted(pack.items())
                ) + f" every={next(iter(pack.values())).move_interval}"
                if pack
                else "spawn-failed" if env.snail_spawn_failed
                else "off"
            )
            print(f"[SNAIL eval] env{env.idx}={snail_state}", flush=True)
        was_training = self.model.training
        self.model.eval()

        try:
            ep_steps = 1
            while not all(agent.done for agent in eval_agents):
                active_agents = [a for a in eval_agents if not a.done]
                if not active_agents:
                    break
                    
                view_tensors = torch.stack([a.prepare_view_tensor() for a in active_agents], dim=0).to(Config.DEVICE)
                state_vectors = torch.stack([a.prepare_state_vector_sequence() for a in active_agents], dim=0).to(Config.DEVICE)
                pos_tensors = torch.stack([a.prepare_pos_sequence() for a in active_agents], dim=0).to(Config.DEVICE)

                with torch.inference_mode():
                    action_logits, direction_logits, _ = self.model(
                        view_tensors, state_vectors, pos_tensors
                    )

                choices = []
                for i, agent in enumerate(active_agents):
                    choices.append((agent, agent.sample_action_choice(
                        action_logits[i],
                        direction_logits[i],
                        action_mask=agent.get_action_mask(),
                        deterministic=True,
                    )))

                offset = (max(1, int(ep_steps)) - 1) % len(choices)
                for agent, choice in choices[offset:] + choices[:offset]:
                    agent.apply_action_choice(choice)

                self._apply_competitive_clear_rewards(
                    [],
                    agents=eval_agents,
                    envs=eval_envs,
                    patch_stored_transitions=False,
                )
                self._synchronize_environment_terminals(
                    agents=eval_agents,
                    envs=eval_envs,
                )
                for env in eval_envs:
                    self.update_environment_for_agents(
                        env,
                        ep_steps,
                        agents=eval_agents,
                    )

                for agent in eval_agents:
                    if getattr(agent, "participating", True):
                        agent.finalize_environment_effects()
                self._synchronize_environment_terminals(
                    agents=eval_agents,
                    envs=eval_envs,
                )
                
                ep_steps += 1

            eps_food = sum(agent.food_eaten for agent in eval_agents)
            eps_steps = max(1, sum(agent.steps for agent in eval_agents))
            eval_rate = eps_food / eps_steps
            
            print(f"Eval Performance: Food Rate={eval_rate:.5f}")
            for idx, agent in enumerate(eval_agents):
                if not getattr(agent, "participating", True):
                    continue
                snail = eval_envs[agent.env_idx].snail_pursuers.get(int(agent.agent_idx))
                print(
                    f"  Eval Agent {idx}: Reward={agent.total_reward:.3f}, "
                    f"Food={agent.food_eaten}, Steps={agent.steps}, "
                    f"Snail={'on' if snail is not None else 'off'}, "
                    f"SnailMoves={snail.moves_made if snail is not None else 0}, "
                    f"SnailDeath={int(agent.killed_by_snail)}\n"
                    f"    {agent.choice_histogram()}"
                )
            print("---------------------------------------\n")
            
            if (
                not getattr(self, "_multi_eval_active", False)
                and eval_rate > self.best_eval_reward
            ):
                print(f"New best evaluation rate! {self.best_eval_reward:.5f} -> {eval_rate:.5f}")
                self.best_eval_reward = eval_rate
                best_path = f"models/ppo_model_{Config.MODEL_NAME}_best.pth"
                # Model-free runs still track the best rate, but have no weights
                # or optimizer state to persist -- a checkpoint would be unloadable.
                if not getattr(Config, "RANDOM_ACTION_POLICY", False):
                    best_checkpoint = self._checkpoint_payload(episode)
                    # Evaluation runs under a temporary RNG stream. A resumed best
                    # checkpoint must continue the training stream from before eval.
                    best_checkpoint['rng_state_torch'] = rng_state_torch
                    best_checkpoint['rng_state_cuda'] = rng_state_cuda
                    best_checkpoint['rng_state_np'] = rng_state_np
                    best_checkpoint['rng_state_py'] = rng_state_py
                    self._atomic_torch_save(
                        best_checkpoint,
                        best_path,
                    )
                    print(f"Saved best model to {best_path}")

        finally:
            self.model.train(was_training)
            
            # Restore RNG states
            torch.set_rng_state(rng_state_torch)
            if rng_state_cuda is not None:
                torch.cuda.set_rng_state(rng_state_cuda)
            np.random.set_state(rng_state_np)
            random.setstate(rng_state_py)
            self.model.release_last_forward_graph()
            del eval_agents, eval_envs
        return eval_rate

    def run_episode(self, stop_event, snapshot_q, target_render_fps):
        self.assign_curriculum_scenarios(self.envs, evaluation=False)
        for env in self.envs:
            env.reset()
        self.assign_context_lengths(self.envs, evaluation=False)
        # reset overlays each episode
        self._in_update = False
        self._training_overlay_frames = 0
        self.episode_start_time = time.monotonic()
        self.reset_agents(self.agents)
        for env in self.envs:
            pack = env.snail_pursuers
            snail_state = (
                "active " + " ".join(
                    f"a{idx}:pos={snail.position}"
                    for idx, snail in sorted(pack.items())
                ) + f" every={next(iter(pack.values())).move_interval}"
                if pack
                else "spawn-failed" if env.snail_spawn_failed
                else "off"
            )
            print(f"[SNAIL train] env{env.idx}={snail_state}", flush=True)

        ep_steps = 1
        last_snapshot_time = 0.0
        
        if Config.VISUALIZE and snapshot_q is not None:
            self.publish_snapshot(snapshot_q, 0)

        while (not stop_event.is_set()) and (not all(agent.done for agent in self.agents)):
            self.update_agents_and_envs(ep_steps)

            if Config.VISUALIZE and snapshot_q is not None:
                now = time.monotonic()
                target_interval = 1.0 / (target_render_fps if target_render_fps > 0 else 30)
                if (now - last_snapshot_time) >= target_interval:
                    self.publish_snapshot(snapshot_q, ep_steps)
                    last_snapshot_time = now
            ep_steps += 1

        if Config.VISUALIZE and snapshot_q is not None:
            self.publish_snapshot(snapshot_q, ep_steps)

    def update_agents_and_envs(self, steps):
        """Run one joint transition, then store rewards after environment dynamics."""
        active_agents = [agent for agent in self.agents if not agent.done]
        if not active_agents:
            return

        view_tensors = torch.stack([a.prepare_view_tensor() for a in active_agents], dim=0).to(Config.DEVICE)
        state_vectors = torch.stack([a.prepare_state_vector_sequence() for a in active_agents], dim=0).to(Config.DEVICE)
        pos_tensors = torch.stack([a.prepare_pos_sequence() for a in active_agents], dim=0).to(Config.DEVICE)

        self.model.eval()
        with torch.no_grad():
            action_logits, direction_logits, value_outputs = self.model(
                view_tensors, state_vectors, pos_tensors
            )

        records = []
        for i, agent in enumerate(active_agents):
            action_mask = agent.get_action_mask()
            choice = agent.sample_action_choice(
                action_logits[i],
                direction_logits[i],
                action_mask=action_mask,
            )
            records.append({
                "agent": agent,
                "state": (
                    view_tensors[i],
                    state_vectors[i],
                    pos_tensors[i],
                    choice["action_mask"],
                    choice["direction_mask"],
                ),
                "choice": choice,
                "value": value_outputs[i],
            })

        # Every policy choice is sampled from the same board state. Rotate who
        # acts first so contested cells do not structurally favor agent zero.
        offset = (max(1, int(steps)) - 1) % len(records)
        execution_order = records[offset:] + records[:offset]
        for record in execution_order:
            (
                record["action"],
                record["direction"],
                record["a_logp"],
                record["d_logp"],
                record["reward"],
                record["done"],
                _,
            ) = record["agent"].apply_action_choice(record["choice"])
        del view_tensors, state_vectors, pos_tensors

        self._apply_competitive_clear_rewards(records)
        # A final bite terminates the entire shared environment before autonomous
        # dynamics can create an irrelevant post-clear collision.
        self._synchronize_environment_terminals()
        for env in self.envs:
            self.update_environment_for_agents(env, steps)

        # Dynamic events belong to the transition that just produced them.
        for record in records:
            agent = record["agent"]
            record["reward"] += agent.finalize_environment_effects()
        self._synchronize_environment_terminals()
        for record in records:
            agent = record["agent"]
            # The cap now carries an explicit remaining-food terminal penalty,
            # so it is part of the finite-horizon task and must not bootstrap
            # into an imaginary post-cap state.
            record["done"] = agent.done
            self.update_agent(
                agent, record["state"], record["action"], record["direction"],
                record["a_logp"], record["d_logp"], record["reward"],
                record["value"], record["done"]
            )

        episode_boundary = all(agent.done for agent in self.agents)
        if episode_boundary and Config.USE_PPO_UPDATE:
            self.rollout_batch_episodes += 1
            pending_samples = sum(
                len(agent.transitions.transitions) for agent in self.agents
            )
            batch_target = max(
                1, int(getattr(Config, "PPO_EPISODES_PER_UPDATE", 8))
            )
            cap = int(Config.MAX_EPISODE_STEPS)
            if cap <= 0:
                # No step cap: an agent survives on spawn energy plus every bite
                # the board holds, so that total is the real per-episode bound.
                rate = max(1e-9, float(Agent.ENERGY_CONSUMPTION_RATE))
                cap = max(
                    int((float(Config.STARTING_ENERGY)
                         + env.initial_food_count
                         * float(Config.FOOD_ENERGY)) / rate) + 1
                    for env in self.envs
                )
            max_samples = (
                batch_target * cap
                * int(Config.NUM_ENVS)
                * int(Config.NUM_AGENTS)
            )
            if pending_samples > max_samples:
                raise RuntimeError(
                    f"rollout buffer exceeded its bound: {pending_samples} > "
                    f"{max_samples} transitions"
                )
            print(
                f"[ROLLOUT BATCH] episodes={self.rollout_batch_episodes}/"
                f"{batch_target} transitions={pending_samples}/{max_samples} "
                "policy_updates=0",
                flush=True,
            )
            if self.rollout_batch_episodes >= batch_target:
                self._update_rollout_batch(steps)

    def _update_rollout_batch(self, steps: int) -> None:
        """Run one PPO update over complete trajectories from one policy state."""
        self._in_update = True
        if Config.VISUALIZE and self.snapshot_q is not None:
            self.publish_snapshot(self.snapshot_q, steps)
        self.model.train()

        transitions = []
        for agent in self.agents:
            transitions.extend(agent.transitions.transitions)
        if not transitions:
            raise RuntimeError("episode batch reached update with no transitions")

        rollout_samples = len(transitions)
        completed_episodes = self.rollout_batch_episodes
        try:
            out_stats = self.ppo.update(transitions)
        except Exception:
            # Agent buffers remain intact so an interrupted/failed update does
            # not silently throw away the complete frozen-policy rollout.
            self._in_update = False
            raise
        for agent in self.agents:
            agent.transitions.clear_memory()
        a_loss = out_stats.get("action_loss", 0.0)
        d_loss = out_stats.get("direction_loss", 0.0)
        v_loss = out_stats.get("value_loss", 0.0)
        compute_time = out_stats.get("compute_time", 0.0)
        approx_kl = out_stats.get("joint_approx_kl", 0.0)
        clip_frac = out_stats.get("joint_clip_frac", 0.0)
        policy_entropy = out_stats.get("action_entropy", 0.0)
        objective_loss = out_stats.get("objective_loss", 0.0)
        joint_policy_loss = out_stats.get("joint_policy_loss", 0.0)
        entropy_bonus = out_stats.get("entropy_bonus", 0.0)
        direction_entropy = out_stats.get("direction_entropy", 0.0)
        direction_relevant_fraction = out_stats.get(
            "direction_relevant_fraction", 0.0
        )
        accepted_post_step_kl = out_stats.get("accepted_post_step_kl", 0.0)
        rejected_post_step_kl = out_stats.get("rejected_post_step_kl", 0.0)
        aux_loss = out_stats.get("aux_loss", 0.0)
        self.rollout_batch_episodes = 0
        self.last_rollout_batch_samples = rollout_samples
        self.last_rollout_update_episode = int(self.current_episode)
        self._in_update = False
        self._training_overlay_frames = 3
        self.last_losses = (a_loss, d_loss, v_loss)
        self.last_compute_time = compute_time
        self.update_count += 1
        self._report_memory(rollout_samples)
        print(
            f"[PPO EPISODE BATCH] update={self.ppo.update_counter} "
            f"episodes={completed_episodes} samples={rollout_samples} "
            f"minibatch={Config.MINIBATCH_SIZE}",
            flush=True,
        )
        if str(Config.DEVICE).startswith("cpu"):
            throttle_ms = out_stats.get("host_package_throttle_ms", float("nan"))
            temperature_c = out_stats.get("host_temperature_c", float("nan"))
            frequency_mhz = out_stats.get("host_frequency_mhz", float("nan"))
            print(
                "[PPO HOST] "
                f"process={out_stats.get('host_process_seconds', 0.0):.2f}s "
                f"parallel={out_stats.get('host_parallelism', 0.0):.2f}x "
                f"package_throttle={throttle_ms:.0f}ms "
                f"temp={temperature_c:.1f}C freq={frequency_mhz:.0f}MHz",
                flush=True,
            )
        print(
            "[PPO OBJECTIVE backpropagated, transition-weighted] "
            f"total={objective_loss:.6f} joint_policy={joint_policy_loss:.6f} "
            f"value={v_loss:.6f}*0.5 "
            f"entropy_bonus={entropy_bonus:.6f} aux={aux_loss:.6f}",
            flush=True,
        )
        print(
            "[PPO POLICY diagnostics only] "
            f"action_surrogate={a_loss:.6f} direction_surrogate={d_loss:.6f}",
            flush=True,
        )
        print(
            "[PPO POLICY health] "
            f"action_entropy={policy_entropy:.6f} "
            f"direction_entropy={direction_entropy:.6f} "
            f"direction_relevant={direction_relevant_fraction:.4f} "
            f"accepted_post_KL={accepted_post_step_kl:.6f} "
            f"rejected_post_KL={rejected_post_step_kl:.6f}",
            flush=True,
        )
        for agent in self.agents:
            agent.action_loss.append(a_loss)
            agent.direction_loss.append(d_loss)
            agent.value_loss.append(v_loss)
            agent.compute_times.append(compute_time)
            agent.approx_kls.append(approx_kl)
            agent.clip_fracs.append(clip_frac)
            agent.policy_entropies.append(policy_entropy)

    def _synchronize_environment_terminals(
        self,
        *,
        agents: Optional[Sequence[Agent]] = None,
        envs: Optional[Sequence[GridEnvironment]] = None,
    ) -> None:
        """A shared environment ends for every resident when its food is cleared."""
        resident_agents = self.agents if agents is None else agents
        resident_envs = self.envs if envs is None else envs
        for env in resident_envs:
            if env.food_remaining > 0:
                continue
            env.episode_cleared = env.initial_food_count > 0
            for agent in resident_agents:
                if agent.env_idx == env.idx:
                    agent.done = True

    def _apply_competitive_clear_rewards(
        self,
        records,
        *,
        agents: Optional[Sequence[Agent]] = None,
        envs: Optional[Sequence[GridEnvironment]] = None,
        patch_stored_transitions: bool = True,
    ) -> None:
        """Award one food-share-scaled completion pool to each arena winner.

        The full completion reward is multiplied by the leading food share.
        Unique leaders receive that pool; exact ties split it. A previously
        terminated leader is credited on its retained final transition.
        """
        if not bool(
            getattr(Config, "COMPETITIVE_FOOD_SHARE_COMPLETION", False)
        ):
            return
        resident_agents = self.agents if agents is None else agents
        resident_envs = self.envs if envs is None else envs
        records_by_agent = {
            id(record["agent"]): record for record in records
        }
        for env in resident_envs:
            if (
                env.food_remaining > 0
                or env.initial_food_count <= 0
                or env.competitive_completion_awarded
                or not uses_competitive_food_share_completion(env)
            ):
                continue
            residents = [
                agent
                for agent in resident_agents
                if agent.env_idx == env.idx
                and bool(getattr(agent, "participating", True))
            ]
            if not residents:
                raise RuntimeError(
                    f"cleared competitive environment {env.idx} has no residents"
                )
            leading_food = max(int(agent.food_eaten) for agent in residents)
            winners = [
                agent for agent in residents
                if int(agent.food_eaten) == leading_food
            ]
            full_completion = float(residents[0].completion_reward())
            winner_pool = (
                full_completion
                * float(leading_food)
                / float(env.initial_food_count)
            )
            payout = winner_pool / float(len(winners))
            for winner in winners:
                record = records_by_agent.get(id(winner))
                if record is not None:
                    record["reward"] += payout
                elif patch_stored_transitions:
                    winner.transitions.add_reward_to_last_transition(payout)
                winner.last_completion_reward = payout
                winner.last_reward += payout
                winner.total_reward += payout
                winner.reward_ledger["completion"] += payout
            env.competitive_completion_awarded = True
            score = "/".join(
                f"a{agent.agent_idx}:{int(agent.food_eaten)}"
                for agent in residents
            )
            winner_ids = "/".join(
                f"a{agent.agent_idx}" for agent in winners
            )
            print(
                f"[COMPETITIVE CLEAR] env{env.idx} food={score} "
                f"winner_pool={winner_pool:.3f} winners={winner_ids} "
                f"payout={payout:.3f}",
                flush=True,
            )

    def update_agent(self, agent, state, action, direction, action_log_prob, direction_log_prob, reward, value, done):
        if state is not None and Config.USE_PPO_UPDATE:
            # This function expects reward (float) then value (tensor)
            agent.transitions.store_transition(state, action, direction, action_log_prob, direction_log_prob, reward, value, done)
        return

    def update_environment_for_agents(
        self,
        env,
        steps,
        *,
        agents: Optional[Sequence[Agent]] = None,
    ):
        resident_agents = self.agents if agents is None else agents
        env_agents = [
            agent for agent in resident_agents if agent.env_idx == env.idx
        ]
        if env_agents and not all(agent.done for agent in env_agents):
            if env.food_tick_enabled:
                env.update_food(steps)
            if env.bouncing_obstacle_objects:
                env.update_bouncing_obstacles(steps)
            if env.snail_pursuers:
                env.update_snail_pursuer(steps)
        # Render events persist until the next published snapshot consumes them.
        if Config.DEBUG:
            env.validate_state(raise_on_error=True)

    def publish_snapshot(self, snapshot_q, tick):
        env_snaps = []
        training_active = self._in_update or (self._training_overlay_frames > 0)
        for env in self.envs:
            agents_here = [
                agent for agent in self.agents
                if agent.env_idx == env.idx
                and bool(getattr(agent, "participating", True))
            ]
            total_agents = len(agents_here)
            dead_agents = [
                agent for agent in agents_here
                if bool(
                    agent.died
                    or agent.killed_by_agent
                    or agent.killed_by_snail
                )
            ]
            live_agents = [a for a in agents_here if not a.done]
            visited = set()
            if Config.DRAW_AGENT_PATH:
                for agent in agents_here:
                    visited.update(agent.visited_cells)

            dead_positions = tuple((a.x, a.y) for a in dead_agents) if total_agents > 1 else ()
            all_dead = total_agents > 0 and len(dead_agents) == total_agents
            has_live = len(live_agents) > 0
            status = (
                "CLEARED" if env.episode_cleared
                else "ALL DEAD" if all_dead
                else "TRAINING" if training_active and has_live
                else None
            )

            snap = env.make_render_snapshot(tick, consume_events=False)
            env_snaps.append(
                EnvRenderSnapshot(
                    grid=snap.grid,
                    moved_obstacles=snap.moved_obstacles,
                    placed_obstacles=snap.placed_obstacles,
                    visited_cells=tuple(visited),
                    dead_agents=dead_positions,
                    status=status,
                    tick=tick,
                )
            )
        # Build a cheap signature to avoid pushing identical frames
        sig = tuple(
            (
                snap.grid.tobytes(),
                snap.moved_obstacles,
                snap.placed_obstacles,
                snap.visited_cells,
                snap.dead_agents,
                snap.status,
                snap.tick if Config.USE_PANEL else None,
            )
            for snap in env_snaps
        )
        # During training overlay, always push frames so the label stays visible.
        if not training_active and sig == self._last_snapshot_sig:
            return
        snapshot = RenderSnapshot(
            envs=env_snaps,
            cell_size=Config.CELL_SIZE,
            timestamp=time.monotonic(),
            panel_data=self._build_panel_data(tick),
        )
        published = False
        try:
            snapshot_q.put_nowait(snapshot)
            published = True
        except queue.Full:
            try:
                snapshot_q.get_nowait()
            except queue.Empty:
                pass
            try:
                snapshot_q.put_nowait(snapshot)
                published = True
            except queue.Full:
                published = False

        if published:
            if not training_active:
                self._last_snapshot_sig = sig
            for env in self.envs:
                env.moved_obstacles.clear()
                env.placed_obstacles.clear()
            if self._training_overlay_frames > 0:
                self._training_overlay_frames -= 1

    def _build_panel_data(self, step: int) -> PanelData:
        elapsed = time.monotonic() - self.episode_start_time
        env_infos = []
        for env in self.envs:
            agents_here = [
                agent for agent in self.agents
                if agent.env_idx == env.idx
                and bool(getattr(agent, "participating", True))
            ]
            live = sum(1 for a in agents_here if not a.done)
            dead = sum(
                1 for agent in agents_here
                if agent.died or agent.killed_by_agent or agent.killed_by_snail
            )
            env_infos.append(
                {
                    "idx": env.idx,
                    "live": live,
                    "dead": dead,
                    "food": getattr(env, "food_remaining", 0),
                    "tick_speed": getattr(env, "food_tick_speed", 0),
                    "variant": env.episode_variant.value,
                }
            )
        return {
            "episode": self.current_episode,
            "step": step,
            "elapsed_s": elapsed,
            "update_count": self.update_count,
            "last_losses": self.last_losses,
            "last_compute": self.last_compute_time,
            "device": Config.DEVICE,
            "envs": env_infos,
        }

    def print_episode_header(self, episode):
        print("-" * 20)
        print(f"Episode {episode}:")
        print("-" * 20)

    def print_episode_summary(self):
        print("Episode Summary:")
        
        eps_food = sum(agent.food_eaten for agent in self.agents)
        eps_steps = max(1, sum(agent.steps for agent in self.agents))
        current_rate = eps_food / eps_steps
        
        current_lr = self.ppo.optimizer.param_groups[0]["lr"]
        print(
            f"PPO Params: GAMMA={self.ppo.gamma:.4f} | "
            f"TAU={self.ppo.tau:.4f} | LR={current_lr:.6f}"
        )
        print(
            "Episode Variant: "
            + " | ".join(
                f"env{env.idx}={env.episode_variant.value}"
                for env in self.envs
            )
        )
        print(f"Performance: Food Rate={current_rate:.5f}")
        batch_target = max(
            1, int(getattr(Config, "PPO_EPISODES_PER_UPDATE", 8))
        )
        if self.last_rollout_update_episode == int(self.current_episode):
            print(
                f"Rollout Batch: updated on {self.last_rollout_batch_samples} "
                f"transitions from {batch_target} frozen-policy episodes"
            )
        else:
            pending_samples = sum(
                len(agent.transitions.transitions) for agent in self.agents
            )
            print(
                f"Rollout Batch: pending {self.rollout_batch_episodes}/"
                f"{batch_target} episodes, {pending_samples} transitions"
            )

        if Config.PPO_TOGETHER:
            total_action_loss = sum(sum(agent.action_loss) for agent in self.agents)
            total_direction_loss = sum(sum(agent.direction_loss) for agent in self.agents)
            total_value_loss = sum(sum(agent.value_loss) for agent in self.agents)
            num_updates = sum(len(agent.action_loss) for agent in self.agents)

            mean_action_loss = total_action_loss / num_updates if num_updates > 0 else 0
            mean_direction_loss = total_direction_loss / num_updates if num_updates > 0 else 0
            mean_value_loss = total_value_loss / num_updates if num_updates > 0 else 0
            
            total_compute_time = sum(self.agents[0].compute_times) if self.agents else 0.0
            print(
                f"Grouped PPO: envs={len(self.envs)} agents={len(self.agents)} "
                f"updates={len(self.agents[0].action_loss) if self.agents else 0} "
                f"compute={total_compute_time:.2f}s"
            )
            print(
                "Global PPO Diagnostics: "
                f"Action Surrogate={mean_action_loss:.4f}, "
                f"Direction Surrogate={mean_direction_loss:.4f}, "
                f"Value Loss={mean_value_loss:.4f}"
            )

            for agent in self.agents:
                if not getattr(agent, "participating", True):
                    continue
                env = self.envs[agent.env_idx]
                scenario = env.curriculum_scenario
                curriculum_label = (
                    f"L{scenario.difficulty} {scenario.label}"
                    if scenario is not None
                    else "global curriculum"
                )
                ledger_str = ", ".join([f"{k}:{v:.2f}" for k, v in agent.reward_ledger.items() if v != 0])
                snail = env.snail_pursuers.get(int(agent.agent_idx))
                print(
                    f"  Env {agent.env_idx}/Agent {agent.agent_idx}: "
                    f"{curriculum_label} | "
                    f"Reward={agent.total_reward:.3f}, Food={agent.food_eaten}, "
                    f"Steps={agent.steps}, Snail={'on' if snail is not None else 'off'}, "
                    f"SnailMoves={snail.moves_made if snail is not None else 0}, "
                    f"SnailDeath={int(agent.killed_by_snail)}, "
                    f"Kills={agent.kills}, KilledByAgent={int(agent.killed_by_agent)}\n"
                    f"    {agent.choice_histogram()}\n"
                    f"    Ledger: {ledger_str}"
                )
                if agent.food_reward_multipliers:
                    print(
                        "    Food Chain: "
                        f"mean={np.mean(agent.food_reward_multipliers):.3f}, "
                        f"max={max(agent.food_reward_multipliers):.3f}, "
                        f"events={len(agent.food_reward_multipliers)}"
                    )
        else:
            for idx, agent in enumerate(self.agents):
                if not getattr(agent, 'participating', True):
                    continue
                mean_action_loss = np.mean(agent.action_loss) if agent.action_loss else 0
                mean_direction_loss = np.mean(agent.direction_loss) if agent.direction_loss else 0
                mean_value_loss = np.mean(agent.value_loss) if agent.value_loss else 0
                mean_approx_kl = np.mean(getattr(agent, "approx_kls", [0.0])) if getattr(agent, "approx_kls", []) else 0
                mean_clip_frac = np.mean(getattr(agent, "clip_fracs", [0.0])) if getattr(agent, "clip_fracs", []) else 0
                mean_policy_entropy = np.mean(getattr(agent, "policy_entropies", [0.0])) if getattr(agent, "policy_entropies", []) else 0
                total_compute_time = sum(agent.compute_times)
                ledger_str = ", ".join([f"{k}:{v:.2f}" for k, v in agent.reward_ledger.items() if v != 0])
                env = self.envs[agent.env_idx]
                snail = env.snail_pursuers.get(int(agent.agent_idx))
                print(
                    f"Agent {idx}: Reward={agent.total_reward:.3f}, Food={agent.food_eaten}, "
                    f"Steps={agent.steps}, Snail={'on' if snail is not None else 'off'}, "
                    f"SnailMoves={snail.moves_made if snail is not None else 0}, "
                    f"SnailDeath={int(agent.killed_by_snail)}, "
                    f"Kills={agent.kills}, KilledByAgent={int(agent.killed_by_agent)}, "
                    f"Action Surrogate={mean_action_loss:.4f}, "
                    f"Direction Surrogate={mean_direction_loss:.4f}, Value Loss={mean_value_loss:.4f}, "
                    f"Approx KL={mean_approx_kl:.4f}, Clip Frac={mean_clip_frac:.4f}, Entropy={mean_policy_entropy:.4f}, "
                    f"Compute Time={total_compute_time:.2f}\n"
                    f"  {agent.choice_histogram()}\n"
                    f"  Ledger: {ledger_str}"
                )
                if agent.food_reward_multipliers:
                    print(
                        "  Food Chain: "
                        f"mean={np.mean(agent.food_reward_multipliers):.3f}, "
                        f"max={max(agent.food_reward_multipliers):.3f}, "
                        f"events={len(agent.food_reward_multipliers)}"
                    )

        if hasattr(self, 'model') and getattr(self.model, 'last_aux_dict', None) is not None:
            aux = self.model.last_aux_dict
            keys_to_print = {
                "lti_rho_budget": "LTI Rho Budget",
                "lti_carry_stack_ratio": "LTI Carry/Stack",
                "lti_carry_rms": "LTI Carry RMS",
                "lti_anchor_drive_rms": "Anchor Drive RMS",
                "lti_stack_output_rms": "Stack Output RMS",
                "lti_anchor_gain_mean": "Anchor Gain",
                "carry_router_bias_ratio": "Carry Router Bias Ratio",
                "union_selected_mean": "Union Selected",
                "union_executed_mean": "Union Executed",
                "step_experts_mean": "Step Executed",
                "union_efficiency": "Union Efficiency",
                "incremental_novelty_mean": "Step Novelty",
                "temporal_axis_price_mean": "Temporal Price",
                "instep_axis_price_mean": "In-Step Price",
                "head_gain_mean": "Head Gain",
                "head_gain_min": "Head Gain Min",
                "head_gain_max": "Head Gain Max",
                "head_basis_orthogonality_rms": "Head Orthog RMS",
                "head_basis_orthogonality_max": "Head Orthog Max",
                "head_basis_condition_number": "Head Basis Condition",
                "router_explore": "Router Explore",
                "router_axis_price_x_max": "Axis Price X Max",
                "router_axis_price_y_max": "Axis Price Y Max",
                "router_axis_overflow": "Axis Overflow",
                "cap_pressure": "Cap Pressure",
                "drop": "Cap Drop",
                "reuse_mass_frac": "Reuse Mass Frac",
                "shared_pool_support": "Shared Pool Support",
                "shared_pool_entropy_fraction": "Shared Pool Entropy",
                "shared_pool_max_fraction": "Shared Pool Max",
                "routed_output_rms": "Routed RMS",
                "shared_post_gate_rms": "Shared RMS",
                "reuse_core_support": "Reuse Core Support",
                "specialist_core_support": "Specialist Core Support",
                "head_selected_collision_frac": "Head Sel Collision",
                "head_executed_collision_frac": "Head Exec Collision",
                "pool_output_rms": "Pool RMS",
                "route_magnitude_sum_mean": "Route Mass",
                "route_magnitude_sum_std": "Route Mass Std",
                "route_magnitude_sum_p95": "Route Mass P95",
                "route_magnitude_l2_mean": "Route L2",
                "route_magnitude_near_zero_frac": "Route Near Zero",
                "route_magnitude_near_one_frac": "Route Near One",
                "route_score_mean": "Route Score Mean",
                "route_score_std": "Route Score Std",
                "route_score_token_mean_std": "Route Level Std",
                "route_score_mass_corr": "Score/Mass Corr",
                "atlas_activation_rms": "Atlas Act RMS",
                "atlas_condition_delta_rms": "Atlas Bend RMS",
                "atlas_selected_base_rms": "Atlas Base RMS",
                "atlas_condition_to_base_ratio": "Atlas Bend/Base",
                "valid_tokens_mean": "Context Tokens",
                "readout_expected_age": "Readout Age Actor/Critic",
                "readout_newest_mass": "Readout Latest Actor/Critic",
                "readout_entropy_fraction": "Readout Entropy Actor/Critic",
                "readout_attention_overlap": "Readout Overlap Actor/Critic",
                "readout_actor_critic_state_cosine": "Readout State Cosine",
            }
            msg_parts = []
            for k, display_name in keys_to_print.items():
                if k in aux:
                    val = aux[k]
                    if k in ("union_selected_mean", "union_executed_mean", "step_experts_mean", "reuse_core_support", "specialist_core_support", "valid_tokens_mean"):
                        msg_parts.append(f"{display_name}={val:.1f}")
                    elif k.startswith("readout_") and hasattr(val, "numel"):
                        values = val.detach().float().reshape(-1).cpu().tolist()
                        msg_parts.append(
                            f"{display_name}=" + "/".join(f"{item:.3f}" for item in values)
                        )
                    elif hasattr(val, 'numel'):
                        val_scalar = val.item() if val.numel() == 1 else val.float().mean().item()
                        msg_parts.append(f"{display_name}={val_scalar:.4f}")
                    elif isinstance(val, float):
                        msg_parts.append(f"{display_name}={val:.4f}")
                    else:
                        msg_parts.append(f"{display_name}={val}")
            if msg_parts:
                print("  Model Aux: " + ", ".join(msg_parts))

        # Mitigation for unbounded list growth
        for agent in self.agents:
            agent.action_loss.clear()
            agent.direction_loss.clear()
            agent.value_loss.clear()
            agent.compute_times.clear()
            agent.approx_kls.clear()
            agent.clip_fracs.clear()
            agent.policy_entropies.clear()

    def should_save_model(self, episode):
        return (
            self.last_rollout_update_episode == int(episode)
            and self.rollout_batch_episodes == 0
            and episode % max(1, int(Config.SAVE_FREQUENCY)) == 0
        )

    @staticmethod
    def _atomic_torch_save(payload, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = f"{path}.tmp-{os.getpid()}"
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _refresh_latest_checkpoint(self, snapshot_path: str) -> None:
        latest_path = f"models/ppo_model_{Config.MODEL_NAME}_latest.pth"
        if os.path.lexists(latest_path):
            os.remove(latest_path)
        try:
            os.link(snapshot_path, latest_path)
        except OSError:
            # Same-filesystem hard links avoid duplicate storage in normal use.
            checkpoint = torch.load(
                snapshot_path, map_location="cpu", weights_only=False
            )
            self._atomic_torch_save(checkpoint, latest_path)
            del checkpoint

    def _prune_periodic_checkpoints(self) -> None:
        keep = max(1, int(getattr(Config, "CHECKPOINT_RETENTION", 3)))
        prefix = f"ppo_model_{Config.MODEL_NAME}_"
        snapshots = []
        for filename in os.listdir("models"):
            if not filename.startswith(prefix) or not filename.endswith(".pth"):
                continue
            suffix = filename[len(prefix):-4]
            if suffix.isdigit():
                snapshots.append((int(suffix), os.path.join("models", filename)))
        snapshots.sort(reverse=True)
        for _, stale_path in snapshots[keep:]:
            os.remove(stale_path)

    def _checkpoint_payload(self, episode: int) -> dict:
        return {
            'architecture_version': getattr(self.model.agent_model, 'architecture_version', 'NavGrid-MLP-v1.0'),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.ppo.optimizer.state_dict(),
            'ppo_update_counter': self.ppo.update_counter,
            'curriculum_idx': self.curriculum_idx,
            'curriculum_level': self.curriculum_level,
            'config_mastered': self.config_mastered,
            'curriculum_episode_counter': self.curriculum_episode_counter,
            'next_episode': int(episode) + 1,
            'episode_boundary_complete': True,
            'env_scale_idx': int(self.env_scale_idx),
            'pending_scale': self._pending_scale,
            'scale_complete': bool(self._scale_complete),
            'best_eval_reward': float(self.best_eval_reward),
            'last_eval_update': int(self.last_eval_update),
            'eval_regression_streak': int(self.eval_regression_streak),
            'curriculum_history': [
                list(history) for history in self.curriculum_history
            ],
            'rng_state_torch': torch.get_rng_state(),
            'rng_state_cuda': (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            ),
            'rng_state_np': np.random.get_state(),
            'rng_state_py': random.getstate(),
        }

    def save_checkpoint(self, episode):
        if getattr(Config, "RANDOM_ACTION_POLICY", False):
            return
        os.makedirs("models", exist_ok=True)
        path = f"models/ppo_model_{Config.MODEL_NAME}_{episode}.pth"
        checkpoint = self._checkpoint_payload(episode)
        self._atomic_torch_save(checkpoint, path)
        self._refresh_latest_checkpoint(path)
        self._prune_periodic_checkpoints()
        print(f"Saved full checkpoint to {path}")
        
    def load_checkpoint(self, path):
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return
            
        checkpoint = torch.load(
            path, map_location=Config.DEVICE, weights_only=False
        )
        if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
            raise TypeError("NavGrid requires a valid checkpoint")
        expected = self.model.agent_model.architecture_version
        actual = checkpoint.get("architecture_version")
        if actual != expected:
            raise ValueError(
                f"Refusing checkpoint architecture {actual!r}; expected {expected!r}"
            )

        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        if 'optimizer_state_dict' not in checkpoint:
            raise KeyError("V32 resume checkpoint has no optimizer state")
        self.ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.ppo.update_counter = int(checkpoint.get('ppo_update_counter', 0))
        self.curriculum_idx = checkpoint.get('curriculum_idx', self.curriculum_idx)
        self.curriculum_level = checkpoint.get('curriculum_level', self.curriculum_level)
        restored_mastered = list(
            checkpoint.get('config_mastered', self.config_mastered)
        )
        # Same padding contract as curriculum_history: new scenarios start unmastered.
        while len(restored_mastered) < len(self.CURRICULUM_SCENARIOS):
            restored_mastered.append(False)
        self.config_mastered = restored_mastered[:len(self.CURRICULUM_SCENARIOS)]
        self.curriculum_episode_counter = checkpoint.get(
            'curriculum_episode_counter', self.curriculum_episode_counter
        )
        self.current_episode = max(1, int(checkpoint.get('next_episode', 1)))
        self._legacy_resume_boundary_pending = not bool(
            checkpoint.get('episode_boundary_complete', False)
        )
        self.env_scale_idx = int(
            checkpoint.get('env_scale_idx', self.env_scale_idx)
        )
        pending_scale = checkpoint.get('pending_scale', self._pending_scale)
        self._pending_scale = (
            tuple(int(value) for value in pending_scale)
            if pending_scale is not None
            else None
        )
        self._scale_complete = bool(
            checkpoint.get('scale_complete', self._scale_complete)
        )
        self.best_eval_reward = float(
            checkpoint.get('best_eval_reward', self.best_eval_reward)
        )
        self.last_eval_update = int(
            checkpoint.get('last_eval_update', self.last_eval_update)
        )
        self.eval_regression_streak = int(
            checkpoint.get(
                'eval_regression_streak', self.eval_regression_streak
            )
        )
        if 'curriculum_history' in checkpoint:
            maxlen = max(1, int(Config.CURRICULUM_WINDOW))
            self.curriculum_history = [
                deque(history, maxlen=maxlen)
                for history in checkpoint['curriculum_history']
            ]
            # Scenarios added after this checkpoint was written have no history.
            # Pad rather than adopt the shorter list, or every index >= the old
            # scenario count raises IndexError on the first draw.
            while len(self.curriculum_history) < len(self.CURRICULUM_SCENARIOS):
                self.curriculum_history.append(deque(maxlen=maxlen))
            del self.curriculum_history[len(self.CURRICULUM_SCENARIOS):]
        if 'rng_state_torch' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state_torch'])
        if checkpoint.get('rng_state_cuda') is not None:
            torch.cuda.set_rng_state(checkpoint['rng_state_cuda'])
        if 'rng_state_np' in checkpoint:
            np.random.set_state(checkpoint['rng_state_np'])
        if 'rng_state_py' in checkpoint:
            random.setstate(checkpoint['rng_state_py'])
        print(f"Loaded exact V32 checkpoint from {path}")


