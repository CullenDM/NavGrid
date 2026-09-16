# NavGrid

**NavGrid** is a 2D multi-agent navigation reinforcement learning environment and benchmark featuring dynamic obstacles, predator pursuers, flocking food swarms, metabolic energy constraints, and an action space that includes a WAIT-safe formulation.

This repository provides the NavGrid environment along with an example Multi-Layer Perceptron (MLP) model demonstrating how to connect custom neural networks and train them using Proximal Policy Optimization (PPO).

---

## Key Features

* **WAIT-Safe Mechanics**:
  * Action space: `MOVE (0)`, `GRAB (1)`, `PLACE (2)`, `WAIT (3)`.
  * `WAIT` burns full metabolic energy (`ENERGY_WAIT_FRACTION = 1.0`).
  * Incomplete-task terminal penalties on starvation and step limits (`-10 * remaining_food_fraction`).
  * `WAIT` transitions are excluded from direction loss, direction entropy, and direction KL divergence, normalizing direction quantities solely over direction-bearing steps.
  * Target KL early stopping with exact Adam/parameter rollback on high-divergence epochs.

* **Dynamic World Elements**:
  * **Food Mechanics**: Configurable movement dynamics via `FOOD_MOVEMENT_MODE`: static food, random adjacent walk (`random`), or evasive flocking boids (`boids` via `FoodSwarmController`, where food forms herds and actively steers away from approaching agents), or per-episode `mixed`.
  * **Obstacle Dynamics**: Static walls, pushable moveable obstacles, inventory-grabbable/placeable obstacles, and bouncing dynamic obstacles.
  * **Predation & Pursuers**: Lethal A* pathfinding `SnailPursuer` and competitive multi-agent arena variants.

* **Observation Space**:
  * **Egocentric vs. Global Views**: Configurable via switch (`Config.USE_GLOBAL_STATE`, `--global-view`, or `NAVGRID_USE_GLOBAL_STATE=1`):
    * **Egocentric View (Default, `USE_GLOBAL_STATE=False`)**: Centered `(B, T, 11, 11)` local observation window that tracks the agent across the world.
    * **Global View (`USE_GLOBAL_STATE=True`)**: Full-grid observation showing the entire environment, with automatic dimension matching, boundary padding for smaller boards, or configurable mapping (`GLOBAL_VIEW_OVERSIZE_POLICY`) for larger boards.
  * **Agent State Feature Vector**: `(B, T, 18)` containing 10 base scalar telemetry values (energy fraction, food remaining, steps since food, carried inventory, etc.) plus one-hot last action (4) and one-hot last direction (4).
  * **Agent Grid Coordinates**: `(B, T, 2)`.

* **Multi-Environment & Multi-Agent Support**:
  * **Multi-Environment Parallel Rollouts**: Step multiple independent grid environments concurrently during episode generation (`Config.NUM_ENVS`, `--num-envs [N]`, or `NAVGRID_NUM_ENVS=N`). The simulation coordinator aggregates transitions across all active environments in parallel, feeding batched experience into PPO updates.
  * **Multi-Agent Environments**: Populate environments with multiple interacting agents (`Config.NUM_AGENTS`, `--num-agents [N]`, or `NAVGRID_NUM_AGENTS=N`), supporting competitive multi-agent arena variants (`DUAL_AGENT` predation, shared resource foraging, and contest mechanics).

* **Example MLP Model (Integration Reference)**:
  * `NavGridMLPPolicy`: Serves as a reference implementation showing how to wire a custom model into the environment. It catches the two sequential inputs output by NavGrid, flattening $(11 \times 11 + 18) \times T = 139 \times 64 = 8,896$ features into a 2-layer MLP trunk with LayerNorm and ReLU activations, outputting action logits, direction logits, and state values.

---

## Quickstart

### Installation

```bash
git clone https://github.com/CullenDM/NavGrid.git
cd NavGrid
pip install -r requirements.txt
```

### Running Training

To train the example MLP model with live Pygame visualization:
```bash
python run_navigation.py
```

To run in headless mode (e.g. on remote servers):
```bash
python run_navigation.py --headless
```

CLI options:
* `--headless`: Disable Pygame rendering.
* `--device [cpu|cuda]`: Target compute device.
* `--num-envs [N]`: Number of parallel environments for simultaneous rollouts (default 1).
* `--num-agents [N]`: Number of agents per environment (default 1).
* `--episodes [N]`: Episodes collected per PPO update (default 4).
* `--minibatch [N]`: Minibatch size for PPO SGD steps (default 80).
* `--random-policy`: Run model-free mode with uniform legal action sampling (no neural net).
* `--global-view`: Switch from default egocentric observation window to full-grid global view.

### Running Tests

Run the test suite:
```bash
python tests/test_navgrid.py
```

---

## Project Structure

```
NavGrid/
├── navgrid/
│   ├── __init__.py         # Package exports
│   ├── constants.py        # Enums (Action, Direction, CellType, ObstacleChoice)
│   ├── config.py           # Sanitized environment and PPO configuration
│   ├── environment.py      # GridEnvironment, BouncingObstacle, SnailPursuer, FoodSwarmController
│   ├── agent.py            # Agent state, inventory, action execution, transition storage
│   ├── model.py            # Example NavGridMLPPolicy showing how to wire in a model
│   ├── ppo.py              # PPO trainer with WAIT-safe loss masking & KL rollback
│   ├── visualizer.py       # Pygame rendering engine and HUD
│   └── simulation.py       # EnvironmentSimulation multi-env and multi-agent rollout runner
├── tests/
│   └── test_navgrid.py     # Unit and integration test suite
├── run_navigation.py       # Main executable training script
├── pyproject.toml          # Packaging specification
└── requirements.txt        # Runtime dependencies
```
