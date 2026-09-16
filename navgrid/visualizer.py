"""Pygame-based visualization and HUD for NavGrid."""

from __future__ import annotations

import math
import os
import queue
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pygame

from .config import Config
from .constants import Action, CellType, Direction, EpisodeVariant

class EnvRenderSnapshot:
    """Immutable render data for a single environment."""
    grid: np.ndarray
    moved_obstacles: Tuple[Tuple[int, int], ...]
    placed_obstacles: Tuple[Tuple[int, int], ...]
    visited_cells: Tuple[Tuple[int, int], ...] = ()
    dead_agents: Tuple[Tuple[int, int], ...] = ()
    status: Optional[str] = None
    tick: Optional[int] = None

class PanelData(TypedDict):
    episode: int
    step: int
    elapsed_s: float
    update_count: int
    last_losses: Tuple[float, float, float]
    last_compute: float
    device: str
    envs: List[dict]

@dataclass(frozen=True)
class RenderSnapshot:
    """Bundle of environment snapshots plus basic render metadata."""
    envs: List[EnvRenderSnapshot]
    cell_size: int
    timestamp: Optional[float] = None
    panel_data: Optional[PanelData] = None


class GridVisualizer:
    """Main-thread pygame renderer that draws snapshots from the worker."""
    def __init__(self, num_envs, env_size, cell_size=Config.CELL_SIZE):
        self.env_size = env_size
        self.cell_size = cell_size
        self.num_envs = max(1, num_envs)
        self.cols = 1
        self.rows = 1
        self.panel_width = 240 if Config.USE_PANEL else 0
        self.buffers = [None] * self.num_envs
        self.font = None
        self.status_font = None
        self.overlay_font = None
        self.bar_height = 24
        self.target_fps = Config.FPS if Config.USE_FPS else 0
        self.dead_font = None

        pygame.init()
        pygame.font.init()

        icon_path = os.environ.get(
            "NAVGRID_ICON",
            "/usr/local/share/pixmaps/navgrid-icon.png",
        )
        if os.path.isfile(icon_path):
            try:
                pygame.display.set_icon(pygame.image.load(icon_path))
            except pygame.error:
                pass

        self.clock = pygame.time.Clock()

        self._configure_layout(self.num_envs)

        # Define colors for the grid items
        self.colors = {
            CellType.EMPTY: (240, 240, 240),
            CellType.AGENT: (255, 0, 0),
            CellType.FOOD: (0, 255, 0),
            CellType.OBSTACLE: (40, 40, 40),
            CellType.MOVEABLE_OBSTACLE: (90, 90, 90),
            CellType.GRABBABLE_OBSTACLE: (125, 125, 125),
            CellType.BOUNCING_OBSTACLE: (190, 190, 0),
            CellType.LOOPING_OBSTACLE: (0, 134, 134),
            CellType.COUNTER_LOOPING_OBSTACLE: (34, 53, 129),
            CellType.WANDERING_OBSTACLE: (0, 223, 186),
            CellType.PURSUER: (124, 67, 139),
            CellType.BOUNDARY: (0, 0, 0),
        }

    def process_events(self) -> bool:
        """
        Pump pygame events. Return False if the app should terminate.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def draw(self, snapshot: Optional[RenderSnapshot]):
        if snapshot is None:
            return

        # Re-layout on environment count, grid size, or cell-size changes.
        snapshot_env_size = snapshot.envs[0].grid.shape[0] if snapshot.envs else self.env_size
        if (
            len(snapshot.envs) != self.num_envs
            or snapshot_env_size != self.env_size
            or int(snapshot.cell_size) != self.cell_size
        ):
            self._configure_layout(
                len(snapshot.envs), env_size=snapshot_env_size,
                cell_size=int(snapshot.cell_size)
            )

        self.screen.fill(self.colors[CellType.EMPTY])
        if Config.USE_PANEL:
            self._draw_panel(snapshot.panel_data)
        overlay_tasks = []
        for env_index, env_snapshot in enumerate(snapshot.envs):
            grid = env_snapshot.grid
            moved = set(env_snapshot.moved_obstacles)
            placed = set(env_snapshot.placed_obstacles)
            visited = set(env_snapshot.visited_cells)
            tile_col = env_index % self.cols
            tile_row = env_index // self.cols
            origin_x = self.panel_width + tile_col * self.env_size * self.cell_size
            origin_y = tile_row * self.env_size * self.cell_size
            for row in range(grid.shape[0]):
                for col in range(grid.shape[1]):
                    cell_val = None
                    try:
                        cell_val = CellType(grid[row, col])
                        color = self.colors.get(cell_val, (128, 128, 128))
                    except Exception:
                        color = (128, 128, 128)

                    if (
                        cell_val is not None
                        and Config.HIGHLIGHT_MOVED_OBSTACLES
                        and (row, col) in moved
                        and cell_val == CellType.MOVEABLE_OBSTACLE
                    ):
                        color = (255, 20, 90)
                    elif (
                        cell_val is not None
                        and Config.HIGHLIGHT_PLACED_OBSTACLES
                        and (row, col) in placed
                        and cell_val == CellType.GRABBABLE_OBSTACLE
                    ):
                        color = (0, 255, 255)
                    elif Config.DRAW_AGENT_PATH and (row, col) in visited and cell_val == CellType.EMPTY:
                        color = (210, 210, 210)  # slightly darker than empty to show path

                    self._draw_cell(origin_x, origin_y, row, col, color)

            # Draw a border and label to make separate envs visually clear
            pygame.draw.rect(
                self.screen,
                (0, 0, 0),
                (
                    origin_x,
                    origin_y,
                    self.env_size * self.cell_size,
                    self.env_size * self.cell_size,
                ),
                width=1,
            )
            if self.font:
                label = self.font.render(f"Env {env_index}", True, (0, 0, 0))
                self.screen.blit(label, (origin_x + 4, origin_y + 4))
            # Per-agent dead markers
            if env_snapshot.dead_agents and self.dead_font:
                for ax, ay in env_snapshot.dead_agents:
                    text_surf = self.dead_font.render("DEAD", True, (255, 255, 255))
                    text_rect = text_surf.get_rect()
                    text_rect.center = (
                        origin_x + ay * self.cell_size + self.cell_size // 2,
                        origin_y + ax * self.cell_size + self.cell_size // 2,
                    )
                    outline_surf = self.dead_font.render("DEAD", True, (0, 0, 0))
                    overlay_tasks.append((outline_surf, text_surf, text_rect))

            if env_snapshot.status and self.overlay_font:
                should_draw = True
                if env_snapshot.status != "ALL DEAD":
                    # Simple blinking: skip every other 500ms chunk
                    should_draw = ((pygame.time.get_ticks() // 500) % 2) == 0
                if should_draw:
                    text_surf = self.overlay_font.render(env_snapshot.status, True, (255, 255, 255))
                    text_rect = text_surf.get_rect()
                    text_rect.center = (
                        origin_x + (self.env_size * self.cell_size) // 2,
                        origin_y + (self.env_size * self.cell_size) // 2,
                    )
                    # Draw simple outline for readability
                    outline_surf = self.overlay_font.render(env_snapshot.status, True, (0, 0, 0))
                    overlay_tasks.append((outline_surf, text_surf, text_rect))

        # Draw overlays after all cells to avoid clipping across tiles
        for outline_surf, text_surf, text_rect in overlay_tasks:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                o_rect = text_rect.copy()
                o_rect.move_ip(dx, dy)
                self.screen.blit(outline_surf, o_rect)
            self.screen.blit(text_surf, text_rect)

        # Status bar with target and measured FPS
        self._draw_status_bar(self.target_fps, self.clock.get_fps())

        pygame.display.flip()

    def _draw_cell(self, origin_x, origin_y, row, col, color):
        rect = (
            origin_x + col * self.cell_size,
            origin_y + row * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.rect(self.screen, color, rect)

    def _draw_status_bar(self, target_fps: float, actual_fps: float):
        bar_y = self.rows * self.env_size * self.cell_size
        start_x = self.panel_width
        bar_width = self.width - self.panel_width
        pygame.draw.rect(self.screen, (0, 0, 0), (start_x, bar_y, bar_width, self.bar_height))
        if self.status_font:
            left_text = self.status_font.render(f"Target FPS: {target_fps:.1f}", True, (255, 255, 255))
            right_text = self.status_font.render(f"Actual FPS: {actual_fps:.1f}", True, (255, 255, 255))
            self.screen.blit(left_text, (start_x + 6, bar_y + 4))
            self.screen.blit(
                right_text,
                (start_x + bar_width - right_text.get_width() - 6, bar_y + 4),
            )

    def _draw_panel(self, panel_data: Optional[dict]):
        if panel_data is None:
            pygame.draw.rect(self.screen, (30, 30, 30), (0, 0, self.panel_width, self.height))
            return
        panel_h = self.height
        pygame.draw.rect(self.screen, (24, 24, 28), (0, 0, self.panel_width, panel_h))
        pygame.draw.rect(self.screen, (70, 70, 80), (0, 0, self.panel_width, panel_h), width=2)
        y = 8
        line_h = 18
        header_color = (180, 200, 255)
        text_color = (235, 235, 235)

        def blit_line(text, color=text_color):
            nonlocal y
            if y + line_h > panel_h - 8:
                return
            surf = self.status_font.render(text, True, color)
            self.screen.blit(surf, (8, y))
            y += line_h
        blit_line("Run", header_color); y += 2
        blit_line(f"Episode {panel_data.get('episode', 0)}")
        blit_line(f"Step {panel_data.get('step', 0)}  Elap {panel_data.get('elapsed_s', 0):.1f}s")
        blit_line(f"Updates {panel_data.get('update_count', 0)}")
        losses = panel_data.get('last_losses', (0, 0, 0))
        blit_line(f"Loss A/D/V {losses[0]:.3f}/{losses[1]:.3f}/{losses[2]:.3f}")
        blit_line(f"Compute {panel_data.get('last_compute', 0):.3f}s")
        blit_line(f"Device {panel_data.get('device', '')}")
        y += 6
        blit_line("Envs", header_color); y += 2
        for env_info in panel_data.get('envs', []):
            blit_line(f"Env {env_info['idx']}  Live:{env_info['live']} Dead:{env_info['dead']}")
            blit_line(f"Food:{env_info['food']}  Tick:{env_info['tick_speed']}")
            blit_line(f"Mode:{env_info.get('variant', 'standard')}")

    def _configure_layout(self, num_envs: int, env_size: Optional[int] = None, cell_size: Optional[int] = None):
        """Choose a near-square layout and resize for current snapshot geometry."""
        if env_size is not None:
            self.env_size = int(env_size)
        if cell_size is not None:
            self.cell_size = int(cell_size)
        self.num_envs = max(1, num_envs)
        requested_columns = int(getattr(Config, "ENV_LAYOUT_COLUMNS", 0))
        self.cols = (
            min(self.num_envs, requested_columns)
            if requested_columns > 0
            else max(1, math.ceil(math.sqrt(self.num_envs)))
        )
        self.rows = max(1, math.ceil(self.num_envs / self.cols))
        self.width = self.panel_width + self.cols * self.env_size * self.cell_size
        self.height = self.rows * self.env_size * self.cell_size + self.bar_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Grid World Training ({self.num_envs} envs)")
        self.buffers = [None] * self.num_envs
        if self.font is None:
            self.font = pygame.font.Font(None, 16)
        if self.status_font is None:
            self.status_font = pygame.font.Font(None, 18)
        if self.overlay_font is None:
            self.overlay_font = pygame.font.Font(None, 28)
        if self.dead_font is None:
            self.dead_font = pygame.font.Font(None, 20)

    def reset(self):
        self.screen.fill(self.colors[CellType.EMPTY])
        pygame.display.update()

    def close(self):
        """
        Ensure all pygame subsystems are shut down (even on exceptions).
        """
        try:
            pygame.display.quit()
        finally:
            pygame.quit()


def render_mainloop(visualizer: GridVisualizer, snapshot_q: queue.Queue, stop_event: threading.Event):
    """
    Main-thread render loop: pumps events, draws latest snapshot, throttles FPS.
    """
    last_snapshot = None
    target_fps = visualizer.target_fps
    first_draw = True
    while not stop_event.is_set():
        if not visualizer.process_events():
            stop_event.set()
            break

        try:
            while True:
                last_snapshot = snapshot_q.get_nowait()
        except queue.Empty:
            pass

        if last_snapshot is not None:
            visualizer.draw(last_snapshot)
            if first_draw:
                first_draw = False

        visualizer.clock.tick(target_fps if target_fps > 0 else 0)
    
