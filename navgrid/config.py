"""Configuration for NavGrid environment, rendering, and PPO training."""

import os
import torch
from .constants import ObstacleSetupChoice

class Config:
    # Energy
    FOOD_ENERGY = 50.0

    # Food Boids Dynamics
    FOOD_BOID_AGENT_AVOID_RADIUS = 8
    FOOD_BOID_NEIGHBOR_RADIUS = 4
    FOOD_BOID_SEPARATION_RADIUS = 3
    FOOD_BOID_AGENT_WEIGHT = 25.0
    FOOD_BOID_COHESION_WEIGHT = 0.35
    FOOD_BOID_COHESION_TARGET_RADIUS = 2.25
    FOOD_BOID_SEPARATION_WEIGHT = 3.2
    FOOD_BOID_ALIGNMENT_WEIGHT = 0.45
    FOOD_BOID_MOMENTUM_WEIGHT = 0.20
    FOOD_BOID_WALL_WEIGHT = 1.65
    FOOD_BOID_WALL_MARGIN = 2
    FOOD_BOID_RANDOM_WEIGHT = 0.10

    # Food Chains & Multipliers
    FOOD_CHAIN_MIN_MULTIPLIER = 1.0
    FOOD_CHAIN_MAX_MULTIPLIER = 2.0
    USE_STEP_MULTIPLIER = False

    # Visualizer options
    USE_FPS = True
    USE_PANEL = False
    DRAW_AGENT_PATH = True
    HIGHLIGHT_MOVED_OBSTACLES = True
    HIGHLIGHT_PLACED_OBSTACLES = True
    SAVE_FREQUENCY = 5

    # GAE Parameter
    TAU = 0.95

    """Sanitized global configuration for NavGrid environment, simulation, and example MLP model."""

    # Environment layout
    NUM_ENVS = int(os.environ.get("NAVGRID_NUM_ENVS", "1"))
    ENVIRONMENT_SIZE = int(os.environ.get("NAVGRID_ENVIRONMENT_SIZE", "11"))
    GRID_SIZE = int(os.environ.get("NAVGRID_VIEW_SIZE", "11"))
    CELL_SIZE = int(os.environ.get("NAVGRID_CELL_SIZE", "15"))
    MIN_EMPTY_PERCENTAGE = 0.5

    # Item counts and dynamics
    VARIABLE_OBSTACLE_COUNT = False
    SET_FOOD_COUNT = int(os.environ.get("NAVGRID_SET_FOOD_COUNT", "10"))
    VARIABLE_FOOD_COUNT = False
    SCALE_FOOD_COUNT = True
    FOOD_TICK = os.environ.get("NAVGRID_FOOD_TICK", "0") == "1"
    USE_RANDOM_FOOD_TICK = False
    FOOD_TICK_SPEED = 4
    MAX_FOOD_TICK_SPEED = 4
    FOOD_MOVEMENT_MODE = os.environ.get("NAVGRID_FOOD_MOVEMENT_MODE", "mixed")

    # Obstacle ecosystem
    OBSTACLE_CHOICE = ObstacleSetupChoice.MIXED_ALL_TYPES
    PRESERVE_PUSH_LANES_FROM_MOVING_FOOD = True

    # Snail Pursuer (Lethal hunting predator)
    USE_IMMORTAL_SNAIL = os.environ.get("NAVGRID_USE_SNAIL", "0") == "1"
    IMMORTAL_SNAIL_EPISODE_PROBABILITY = 0.10
    IMMORTAL_SNAIL_MOVE_INTERVAL = 5
    IMMORTAL_SNAIL_MIN_SPAWN_DISTANCE = 5
    IMMORTAL_SNAIL_MAX_SPAWN_DISTANCE = 20
    IMMORTAL_SNAIL_GENERATION_ATTEMPTS = 20
    REQUIRE_SNAIL_ON_SNAIL_EPISODE = True
    IMMORTAL_SNAIL_DEATH_REWARD = -10.0

    # Multi-Agent Arena Variants
    ARENA_VARIANTS_ENABLED = os.environ.get("NAVGRID_ARENA_VARIANTS", "0") == "1"
    EVALUATION_ARENA_VARIANT = "standard"
    USE_AGENT_PREDATION = False
    AGENT_VERSUS_AGENT_EPISODE_PROBABILITY = 0.10
    COMPETITIVE_FOOD_SHARE_COMPLETION = False

    # Energy & Metabolism Model (WAIT burns full energy tick)
    STARTING_ENERGY = float(os.environ.get("NAVGRID_STARTING_ENERGY", "100.0"))
    ENERGY_FOR_FOOD = float(os.environ.get("NAVGRID_FOOD_ENERGY", "100.0"))
    ENERGY_CONSUMPTION_RATE = 1.0
    ENERGY_WAIT_FRACTION = 1.0
    ENERGY_EXTRA_FOR_PUSH_SUCCESS = 0.5
    ENERGY_EXTRA_FOR_PUSH_FAIL = 1.0
    ENERGY_EXTRA_FOR_GRAB_SUCCESS = 0.5
    ENERGY_EXTRA_FOR_GRAB_FAIL = 1.0
    ENERGY_EXTRA_FOR_PLACE_SUCCESS = 0.5
    MAX_INVENTORY = 5

    # Action Space (Canonical: Move, Grab, Place, Wait)
    NUM_ACTIONS = int(os.environ.get("NAVGRID_NUM_ACTIONS", "4"))
    NUM_DIRECTIONS = int(os.environ.get("NAVGRID_NUM_DIRECTIONS", "4"))

    # Observation Layout (10 base scalars + action one-hot + direction one-hot = 18)
    BASE_STATE_DIM = 10 + NUM_ACTIONS + NUM_DIRECTIONS
    STATE_DIM = BASE_STATE_DIM
    SEQUENCE_LENGTH = int(os.environ.get("NAVGRID_SEQUENCE_LENGTH", "64"))
    SEQUENCE_LENGTH_CHOICES = (SEQUENCE_LENGTH,)
    SEQUENCE_LENGTH_WEIGHTS = (1.0,)
    EVAL_SEQUENCE_LENGTH = SEQUENCE_LENGTH
    USE_GLOBAL_STATE = os.environ.get("NAVGRID_USE_GLOBAL_STATE", "0") == "1"
    GLOBAL_VIEW_OVERSIZE_POLICY = "nearest"

    # Evaluation Regression Guards
    EVAL_SEEDS = (42, 314159, 271828)
    EVAL_REGRESSION_PATIENCE = 3
    EVAL_REGRESSION_FRACTION = 0.25
    EVAL_REGRESSION_MIN_BEST_RATE = 0.02

    # Curriculum Learning
    CURRICULUM_ENABLED = os.environ.get("NAVGRID_CURRICULUM", "1") == "1"
    CURRICULUM_ALL_SCENARIOS = False
    CURRICULUM_PER_ENV_MIX = False
    CURRICULUM_WINDOW = 12
    CURRICULUM_MIN_TRIALS = 6
    CURRICULUM_MASTERY_RATE = 0.65
    CURRICULUM_RETENTION_RATE = 0.55
    CURRICULUM_FRONTIER_EPISODES = 3
    CURRICULUM_SCALE_ENABLED = False
    CURRICULUM_ENV_SUCCESS_FRACTION = 1.0
    REACHABILITY_POLICY = "strict"
    REACHABILITY_REGEN_ATTEMPTS = 20

    # Visualization & HUD
    VISUALIZE = os.environ.get("NAVGRID_VISUALIZE", "1") == "1"
    HEADLESS = os.environ.get("NAVGRID_HEADLESS", "0") == "1"
    FPS = int(os.environ.get("NAVGRID_FPS", "60"))
    USE_CONTEXT_FOOD_CHAIN_BONUS = False

    # Example MLP Architecture (Wiring Reference)
    MLP_HIDDEN_DIMS = (256, 128)
    MODEL_NAME = os.environ.get("NAVGRID_MODEL_NAME", "NavGrid-MLP")
    MODEL_PATH = f"./models/ppo_model_{MODEL_NAME}.pth"
    LOAD_MODEL = False
    RANDOM_ACTION_POLICY = False

    # PPO Hyperparameters & WAIT-Safe Optimization
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE = float(os.environ.get("NAVGRID_LR", "1e-4"))
    ACTION_ENTROPY_COEF = 0.0025
    DIRECTION_ENTROPY_COEF = 0.005
    CLIP_PARAM = 0.2
    TARGET_KL = 0.02
    USE_EARLY_STOPPING = True
    PPO_EPISODES_PER_UPDATE = int(os.environ.get("NAVGRID_EPISODES_PER_UPDATE", "4"))
    MINIBATCH_SIZE = int(os.environ.get("NAVGRID_MINIBATCH_SIZE", "80"))
    PPO_EPOCHS = int(os.environ.get("NAVGRID_PPO_EPOCHS", "4"))
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    USE_PPO_UPDATE = True
    MAX_EPISODE_STEPS = int(os.environ.get("NAVGRID_MAX_EPISODE_STEPS", "128"))
    UPDATE_FREQUENCY = 0
    PPO_TOGETHER = False
    NUM_AGENTS = 1

    # Learning Rate Schedule
    LR_WARMUP_UPDATES = 25
    LR_WARMUP_START_FACTOR = 0.25
    LR_PEAK_HOLD_UPDATES = 1000
    COSINE_LR_DECAY_STEPS = 20000
    LR_PERPETUAL_COSINE_RESTARTS = True
    LR_SCHEDULE_MODE = os.environ.get("NAVGRID_LR_SCHEDULE_MODE", "cosine")
    LR_ANNEAL_START_UPDATE = 0
    LR_ANNEAL_DECAY_UPDATES = 5000
    MIN_LEARNING_RATE = 1e-5
    USE_COSINE_LR_DECAY = True
    CHECKPOINT_RETENTION = 3
    GC_INTERVAL_UPDATES = 25
    DEBUG = False
