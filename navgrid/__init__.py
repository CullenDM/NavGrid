"""NavGrid: 2D multi-agent navigation reinforcement learning environment with WAIT-safe PPO."""

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
from .environment import (
    BouncingObstacle,
    CurriculumScenario,
    FoodSwarmController,
    GridEnvironment,
    SnailPursuer,
)
from .agent import Agent, TransitionStorage
from .model import NavGridMLPPolicy, PPOAgent
from .ppo import PPO
from .visualizer import GridVisualizer, render_mainloop
from .simulation import EnvironmentSimulation

__version__ = "1.0.0"

__all__ = [
    "Config",
    "CellType",
    "Action",
    "Direction",
    "EpisodeVariant",
    "ObstacleSetupChoice",
    "FoodMovementMode",
    "CARDINAL_DELTAS",
    "GridEnvironment",
    "CurriculumScenario",
    "BouncingObstacle",
    "SnailPursuer",
    "FoodSwarmController",
    "Agent",
    "TransitionStorage",
    "NavGridMLPPolicy",
    "PPOAgent",
    "PPO",
    "GridVisualizer",
    "render_mainloop",
    "EnvironmentSimulation",
]
