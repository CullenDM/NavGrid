"""Constants and enums for the NavGrid environment."""

from enum import Enum, IntEnum
from typing import Tuple, Dict

class CellType(IntEnum):
    """Cell identifiers used across grids, views, and rendering."""
    BOUNDARY = 1
    WANDERING_OBSTACLE = 2
    COUNTER_LOOPING_OBSTACLE = 3
    LOOPING_OBSTACLE = 4
    BOUNCING_OBSTACLE = 5
    OBSTACLE = 6
    MOVEABLE_OBSTACLE = 7
    GRABBABLE_OBSTACLE = 8
    EMPTY = 9
    FOOD = 10
    AGENT = 11
    OTHER_AGENT = 12
    PURSUER = 13


class EpisodeVariant(str, Enum):
    """Mutually exclusive environment dynamics selected once per episode."""
    STANDARD = "standard"
    IMMORTAL_SNAIL = "immortal_snail"
    DUAL_AGENT = "dual_agent"


class Action(Enum):
    """Available agent actions."""
    MOVE = 0
    GRAB = 1
    PLACE = 2
    WAIT = 3


class Direction(Enum):
    """Cardinal movement directions."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class ObstacleSetupChoice(IntEnum):
    """Preset mixes of obstacle types for environment generation."""
    ONLY_STATIC_LARGE = 0
    ONLY_MOVEABLE_LARGE = 1
    ONLY_GRABBABLE_LARGE = 2
    MIXED_SMALL_NO_BOUNCE = 3
    EMPTY_NO_OBSTACLES = 4
    MIXED_RANDOM_SMALL_NO_BOUNCE = 5
    STATIC_MOVEABLE_HALF = 6
    MOVEABLE_GRABBABLE_HALF = 7
    STATIC_GRABBABLE_HALF = 8
    MIXED_SMALL_BOUNCE_LOW = 9
    MIXED_SMALL_BOUNCE_MED = 10
    MOVEABLE_GRABBABLE_BOUNCE_MED = 11
    STATIC_GRABBABLE_BOUNCE_MED = 12
    STATIC_MOVEABLE_BOUNCE_MED = 13
    ONLY_STATIC_BOUNCE_MED = 14
    ONLY_MOVEABLE_BOUNCE_MED = 15
    ONLY_GRABBABLE_BOUNCE_MED = 16
    MIXED_ALL_TYPES = 17


class FoodMovementMode(Enum):
    """Supported movement policies for food cells."""
    RANDOM = "random"
    BOIDS = "boids"
    MIXED = "mixed"

    @classmethod
    def normalize(cls, value):
        """Return a valid FoodMovementMode, falling back to RANDOM on bad config values."""
        if isinstance(value, cls):
            return value

        raw_value = str(value).strip().lower().replace("-", "_")
        aliases = {
            "rand": cls.RANDOM,
            "random_walk": cls.RANDOM,
            "dumb": cls.RANDOM,
            "boid": cls.BOIDS,
            "flock": cls.BOIDS,
            "flocking": cls.BOIDS,
            "swarm": cls.BOIDS,
            "random_or_boids": cls.MIXED,
            "random_or_boid": cls.MIXED,
            "boids_or_random": cls.MIXED,
            "episode_random": cls.MIXED,
            "per_episode_random": cls.MIXED,
        }
        if raw_value in aliases:
            return aliases[raw_value]

        for mode in cls:
            if raw_value == mode.value:
                return mode

        print(f"Warning: Invalid FoodMovementMode ({value!r}). Defaulting to 'random'.")
        return cls.RANDOM

    @classmethod
    def resolve_episode_mode(cls, value):
        """Resolve the configured food policy to the concrete policy used for this reset."""
        mode = cls.normalize(value)
        if mode == cls.MIXED:
            import random
            return cls.BOIDS if random.random() < 0.5 else cls.RANDOM
        return mode


# One canonical action-direction convention for MOVE, GRAB, and PLACE.
# Indices correspond to Direction.UP (0), DOWN (1), LEFT (2), RIGHT (3).
CARDINAL_DELTAS: Tuple[Tuple[int, int], ...] = (
    (-1, 0),  # UP
    (1, 0),   # DOWN
    (0, -1),  # LEFT
    (0, 1),   # RIGHT
)

DIRECTION_TO_DELTA: Dict[Direction, Tuple[int, int]] = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}
