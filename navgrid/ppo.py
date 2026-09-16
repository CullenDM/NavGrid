"""PPO Trainer with WAIT-safe loss masking and GAE for NavGrid."""

from __future__ import annotations

import copy
import gc
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from .config import Config
from .constants import Action, Direction

class PPO:
    """PPO trainer with optional halting regularization hooks."""
    def __init__(self, model, optimizer, clip_param=0.2, ppo_epochs=1,
                 target_kl=0.01, gamma=0.99, tau=0.95):
        self.model = model
        self.optimizer = optimizer
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.target_kl = target_kl
        self.gamma = gamma
        self.tau = tau
        self.transitions = []

        # cache defaults safely even if Config lacks them
        self.halt_coef = float(getattr(Config, "PPO_HALTING_COEF", 0.0))
        self.ponder_penalty = float(getattr(Config, "PPO_PONDER_PENALTY", 0.0))
        self.sel_ent_coef = float(getattr(Config, "PPO_SELECTOR_ENT_COEF", 0.0))

        # for optional introspection/printing
        self._last_aux = {}
        self.update_counter = 0
        self.forensic_grad_checks = (
            os.environ.get("NAVGRID_PPO_FORENSICS", "0") == "1"
        )

    @staticmethod
    def _host_performance_snapshot() -> dict[str, float]:
        """Read cheap host counters that disambiguate CPU slowdowns."""
        snapshot = {
            "process_seconds": time.process_time(),
            "package_throttle_ms": float("nan"),
            "temperature_c": float("nan"),
            "frequency_mhz": float("nan"),
        }
        if not str(Config.DEVICE).startswith("cpu"):
            return snapshot

        throttle_values = []
        frequency_values = []
        cpu_root = "/sys/devices/system/cpu"
        try:
            cpu_entries = [
                entry.path
                for entry in os.scandir(cpu_root)
                if entry.is_dir() and entry.name.startswith("cpu")
                and entry.name[3:].isdigit()
            ]
        except OSError:
            cpu_entries = []
        for cpu_path in cpu_entries:
            for relative_path, output in (
                ("thermal_throttle/package_throttle_total_time_ms", throttle_values),
                ("cpufreq/scaling_cur_freq", frequency_values),
            ):
                try:
                    with open(
                        os.path.join(cpu_path, relative_path),
                        "r",
                        encoding="ascii",
                    ) as handle:
                        output.append(float(handle.read().strip()))
                except (OSError, ValueError):
                    pass

        temperatures = []
        thermal_root = "/sys/class/thermal"
        try:
            thermal_entries = [
                entry.path
                for entry in os.scandir(thermal_root)
                if entry.is_dir() and entry.name.startswith("thermal_zone")
            ]
        except OSError:
            thermal_entries = []
        for zone_path in thermal_entries:
            try:
                with open(
                    os.path.join(zone_path, "temp"), "r", encoding="ascii"
                ) as handle:
                    temperatures.append(float(handle.read().strip()))
            except (OSError, ValueError):
                pass

        if throttle_values:
            snapshot["package_throttle_ms"] = max(throttle_values)
        if temperatures:
            snapshot["temperature_c"] = max(temperatures) / 1000.0
        if frequency_values:
            snapshot["frequency_mhz"] = (
                sum(frequency_values) / len(frequency_values) / 1000.0
            )
        return snapshot

    @staticmethod
    def _cpu_forensic_copy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, tuple):
            return tuple(PPO._cpu_forensic_copy(item) for item in value)
        if isinstance(value, dict):
            return {
                key: PPO._cpu_forensic_copy(item)
                for key, item in value.items()
            }
        return value

    def _gradient_forensics(self):
        bad = []
        largest = []
        squared_norm = 0.0
        for name, parameter in self.model.named_parameters():
            grad = parameter.grad
            if grad is None:
                continue
            finite = torch.isfinite(grad)
            nonfinite_count = int((~finite).sum().item())
            if nonfinite_count:
                bad.append({
                    "name": name,
                    "shape": tuple(grad.shape),
                    "nonfinite": nonfinite_count,
                    "nan": int(torch.isnan(grad).sum().item()),
                    "posinf": int(torch.isposinf(grad).sum().item()),
                    "neginf": int(torch.isneginf(grad).sum().item()),
                })
                continue
            grad_norm = float(torch.linalg.vector_norm(grad.detach().double()).item())
            squared_norm += grad_norm * grad_norm
            largest.append((grad_norm, name, tuple(grad.shape)))
        largest.sort(reverse=True)
        return bad, math.sqrt(squared_norm), largest[:12]

    def _save_ppo_forensics(self, data, context):
        os.makedirs("models/forensics", exist_ok=True)
        path = os.path.join(
            "models",
            "forensics",
            f"ppo_nonfinite_update_{self.update_counter:06d}.pt",
        )
        payload = {
            "update_counter": int(self.update_counter),
            "optimizer_step": int(self._model_optimizer_step()),
            "context": context,
            "data": self._cpu_forensic_copy(data),
            "model_state_dict": self._cpu_forensic_copy(
                self.model.state_dict()
            ),
            "optimizer_state_dict": self._cpu_forensic_copy(
                self.optimizer.state_dict()
            ),
            "rng_state_torch": torch.get_rng_state(),
            "rng_state_np": np.random.get_state(),
            "rng_state_py": random.getstate(),
        }
        temporary = f"{path}.tmp"
        torch.save(payload, temporary)
        os.replace(temporary, path)
        print(f"[PPO NONFINITE] forensic replay saved to {path}", flush=True)
        return path

    def _model_optimizer_step(self) -> int:
        agent_model = getattr(self.model, "agent_model", None)
        step_count = getattr(agent_model, "optimizer_step_count", None)
        if step_count is None:
            return 0
        try:
            return int(step_count.item())
        except AttributeError:
            return int(step_count)

    def _lr_for_update(self, update: int) -> float:
        peak_lr = float(getattr(Config, "LEARNING_RATE", 3e-4))
        min_lr = float(getattr(Config, "MIN_LEARNING_RATE", 3e-5))
        schedule_mode = str(
            getattr(Config, "LR_SCHEDULE_MODE", "cosine")
        ).strip().lower()
        if schedule_mode == "anchored_linear_floor":
            start_update = max(
                0, int(getattr(Config, "LR_ANNEAL_START_UPDATE", 0))
            )
            decay_updates = max(
                1, int(getattr(Config, "LR_ANNEAL_DECAY_UPDATES", 5000))
            )
            if update <= start_update:
                return peak_lr
            progress = min(
                1.0,
                float(update - start_update) / float(decay_updates),
            )
            return peak_lr + (min_lr - peak_lr) * progress
        if schedule_mode != "cosine":
            raise ValueError(f"unknown LR_SCHEDULE_MODE: {schedule_mode!r}")

        warmup = max(0, int(getattr(Config, "LR_WARMUP_UPDATES", 50)))
        warmup_start_factor = min(
            1.0, max(0.0, float(getattr(Config, "LR_WARMUP_START_FACTOR", 0.25)))
        )
        peak_hold = max(0, int(getattr(Config, "LR_PEAK_HOLD_UPDATES", 0)))
        decay_updates = max(1, int(getattr(Config, "COSINE_LR_DECAY_STEPS", 2000)))

        if warmup > 0 and update <= warmup:
            progress = float(update) / float(warmup)
            return peak_lr * (
                warmup_start_factor + (1.0 - warmup_start_factor) * progress
            )
        if not bool(getattr(Config, "USE_COSINE_LR_DECAY", True)):
            return peak_lr

        post_warmup = max(0, int(update) - warmup - 1)
        cycle_length = peak_hold + decay_updates
        if bool(getattr(Config, "LR_PERPETUAL_COSINE_RESTARTS", False)):
            cycle_step = post_warmup % max(1, cycle_length)
        else:
            cycle_step = min(post_warmup, max(0, cycle_length - 1))
        if cycle_step < peak_hold:
            return peak_lr

        decay_step = cycle_step - peak_hold
        frac = min(1.0, decay_step / float(max(1, decay_updates - 1)))
        return min_lr + 0.5 * (peak_lr - min_lr) * (
            1.0 + math.cos(math.pi * frac)
        )

    def compute_returns_and_advantages(self):
        """
        Generalized Advantage Estimation over *multiple* segments.
    
        Segments are delimited by either:
          - done == True       (true terminal; bootstrap = 0)
          - truncated == True  (rollout cut; bootstrap = provided V(s_{t+1}))
        """
        device = torch.device(Config.DEVICE)
        gamma, tau = self.gamma, self.tau
    
        T = len(self.transitions)
        if T == 0:
            return (torch.empty(0, dtype=torch.float32, device=device),
                    torch.empty(0, dtype=torch.float32, device=device))
    
        rewards, values, dones, truncs, boots = [], [], [], [], []
    
        for item in self.transitions:
            # reward
            r = torch.as_tensor(item[5], dtype=torch.float32, device=device)
            # value V(s_t) — may be a CPU tensor; ensure device move
            v = item[6]
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=torch.float32)
            v = v.to(device).reshape(())
            # done
            d = torch.as_tensor(item[7], dtype=torch.float32, device=device)
    
            # truncation & bootstrap value (V(s_{t+1})) if present
            if len(item) >= 10:
                tflag = torch.as_tensor(1.0 if item[8] else 0.0, dtype=torch.float32, device=device)
                b = item[9]
                if not isinstance(b, torch.Tensor):
                    b = torch.tensor(b, dtype=torch.float32)
                b = b.to(device).reshape(())
            else:
                tflag = torch.tensor(0.0, dtype=torch.float32, device=device)
                b = torch.tensor(0.0, dtype=torch.float32, device=device)
    
            rewards.append(r)
            values.append(v)
            dones.append(d)
            truncs.append(tflag)
            boots.append(b)
    
        rewards = torch.stack(rewards)   # [T]
        values  = torch.stack(values)    # [T]
        dones   = torch.stack(dones)     # [T]
        truncs  = torch.stack(truncs)    # [T]
        boots   = torch.stack(boots)     # [T]  (all on the correct device now)
    
        returns_all = []
        advantages_all = []
    
        start = 0
        while start < T:
            end = start
            while end < T - 1 and (dones[end].item() == 0.0 and truncs[end].item() == 0.0):
                end += 1
    
            seg_rewards = rewards[start:end+1]
            seg_values  = values[start:end+1]
            seg_dones   = dones[start:end+1]
            seg_trunc   = truncs[start:end+1]
            seg_len     = seg_rewards.size(0)
    
            # Determine bootstrap for this segment
            if seg_dones[-1] > 0.5:
                bootstrap = torch.tensor(0.0, dtype=torch.float32, device=device)
            elif seg_trunc[-1] > 0.5:
                bootstrap = boots[start + seg_len - 1].detach()
            else:
                bootstrap = torch.tensor(0.0, dtype=torch.float32, device=device)
    
            values_ext = torch.cat([seg_values, bootstrap.view(1)], dim=0)
    
            gae = torch.tensor(0.0, dtype=torch.float32, device=device)
            seg_returns = []
            for i in reversed(range(seg_len)):
                mask  = 1.0 - seg_dones[i]  # True terminal shuts off both gamma and tau
                delta = seg_rewards[i] + gamma * values_ext[i + 1] * mask - values_ext[i]
                gae   = delta + gamma * tau * mask * gae
                seg_returns.insert(0, gae + values_ext[i])
    
            seg_returns   = torch.stack(seg_returns)
            seg_advantage = seg_returns - seg_values
    
            returns_all.append(seg_returns)
            advantages_all.append(seg_advantage)
    
            start = end + 1
    
        returns_tensor = torch.cat(returns_all, dim=0)
        advantages     = torch.cat(advantages_all, dim=0)
    
        # Advantage normalization (safe)
        mean = advantages.mean()
        std  = advantages.std(unbiased=False)
        if (not torch.isfinite(std)) or (std < 1e-8):
            advantages = advantages - mean
        else:
            advantages = (advantages - mean) / std
    
        returns_tensor = torch.nan_to_num(returns_tensor)
        advantages     = torch.nan_to_num(advantages)
    
        return returns_tensor.detach(), advantages.detach()

    def _aux_halting_terms(self, grid_inputs, state_inputs):
        """
        Safely query model for halting aux terms; returns dict with tensors
        on correct device or None if unsupported.
        """
        if not hasattr(self.model, "get_halting_aux"):
            return None
        aux = self.model.get_halting_aux((grid_inputs, state_inputs))
        if not isinstance(aux, dict) or len(aux) == 0:
            return None
        # ensure device consistency
        out = {}
        for k, v in aux.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(Config.DEVICE)
        return out

    def compute_loss(self, mb):
        # Policy/value losses
        new_a_log, new_d_log, new_vals, a_entropy, d_entropy = self.model.get_new_log_probs(
            mb['state_inputs'],
            mb['actions'],
            mb['directions'],
            mb.get('action_masks', None),
            mb.get('direction_masks', None),
            collect_aux=bool(mb.get('_collect_diagnostics', False)),
        )
        
        valid_mask = mb.get('valid_mask', None)
        if valid_mask is not None:
            valid_mask = valid_mask.float()
            valid_count = valid_mask.sum().clamp_min(1.0)
        else:
            valid_count = float(mb['actions'].size(0))

        # Action Loss
        log_ratio_action = new_a_log - mb['old_action_log_probs']
        ratio_action = log_ratio_action.exp()
        
        action_surr_unclipped = ratio_action * mb['advantages']
        action_surr_clipped = ratio_action.clamp(
            1.0 - self.clip_param,
            1.0 + self.clip_param
        ) * mb['advantages']
        action_surr = torch.minimum(
            action_surr_unclipped,
            action_surr_clipped
        )
        if valid_mask is not None:
            action_loss = -(action_surr * valid_mask).sum() / valid_count
        else:
            action_loss = -action_surr.mean()

        # Direction Loss
        log_ratio_direction = new_d_log - mb['old_direction_log_probs']
        ratio_direction = log_ratio_direction.exp()
        direction_relevant = mb['actions'].ne(Action.WAIT.value)
        if valid_mask is not None:
            direction_relevant = direction_relevant & (valid_mask > 0.5)
        direction_weight = direction_relevant.to(log_ratio_direction.dtype)
        direction_count = direction_weight.sum().clamp_min(1.0)

        def direction_mean(values):
            return (values * direction_weight).sum() / direction_count
        
        direction_surr_unclipped = ratio_direction * mb['advantages']
        direction_surr_clipped = ratio_direction.clamp(
            1.0 - self.clip_param,
            1.0 + self.clip_param
        ) * mb['advantages']
        
        direction_loss = -direction_mean(torch.minimum(
            direction_surr_unclipped,
            direction_surr_clipped
        ))

        # Value loss: SmoothL1 + clip
        v_clipped = mb['old_values'] + torch.clamp(
            new_vals - mb['old_values'], 
            -self.clip_param, 
            self.clip_param
        )
        v_loss_clip = F.smooth_l1_loss(v_clipped, mb['returns'], reduction='none')
        v_loss_plain = F.smooth_l1_loss(new_vals, mb['returns'], reduction='none')
        v_loss_elem = torch.maximum(v_loss_plain, v_loss_clip)
        if valid_mask is not None:
            value_loss = (v_loss_elem * valid_mask).sum() / valid_count
        else:
            value_loss = v_loss_elem.mean()

        # Halt auxiliaries and diagnostics
        aux_loss = torch.tensor(0.0, device=Config.DEVICE)
        
        # WAIT has no direction decision. Keep its PPO ratio action-only and do
        # not let WAIT-heavy batches dilute direction learning or exploration.
        effective_direction_log_ratio = torch.where(
            direction_relevant,
            log_ratio_direction,
            torch.zeros_like(log_ratio_direction),
        )
        joint_log_ratio = log_ratio_action + effective_direction_log_ratio
        joint_ratio = joint_log_ratio.exp()
        joint_surr_unclipped = joint_ratio * mb['advantages']
        joint_surr_clipped = joint_ratio.clamp(
            1.0 - self.clip_param,
            1.0 + self.clip_param,
        ) * mb['advantages']
        joint_surr = torch.minimum(
            joint_surr_unclipped,
            joint_surr_clipped,
        )
        if valid_mask is not None:
            joint_policy_loss = -(joint_surr * valid_mask).sum() / valid_count
        else:
            joint_policy_loss = -joint_surr.mean()
        
        def masked_mean(values):
            if valid_mask is not None:
                return ((values * valid_mask).sum() / valid_count).item()
            return values.mean().item()

        joint_approx_kl = masked_mean((joint_ratio - 1.0) - joint_log_ratio)
        joint_clip_frac = masked_mean(((joint_ratio - 1.0).abs() > self.clip_param).float())
        
        action_approx_kl = masked_mean((ratio_action - 1.0) - log_ratio_action)
        action_clip_frac = masked_mean(((ratio_action - 1.0).abs() > self.clip_param).float())
        
        direction_approx_kl = direction_mean(
            (ratio_direction - 1.0) - log_ratio_direction
        ).item()
        direction_clip_frac = direction_mean(
            ((ratio_direction - 1.0).abs() > self.clip_param).float()
        ).item()
        direction_entropy = direction_mean(d_entropy)
        relevant_direction_ratios = log_ratio_direction[direction_relevant]
        if relevant_direction_ratios.numel() == 0:
            direction_log_ratio_min = 0.0
            direction_log_ratio_max = 0.0
        else:
            direction_log_ratio_min = float(
                relevant_direction_ratios.detach().min().item()
            )
            direction_log_ratio_max = float(
                relevant_direction_ratios.detach().max().item()
            )
        
        if valid_mask is not None:
            valid_bool = valid_mask > 0.5
            valid_act_lr = log_ratio_action[valid_bool]
            valid_jnt_lr = joint_log_ratio[valid_bool]
            action_log_ratio_min = float(valid_act_lr.detach().min().item()) if valid_act_lr.numel() > 0 else 0.0
            action_log_ratio_max = float(valid_act_lr.detach().max().item()) if valid_act_lr.numel() > 0 else 0.0
            joint_log_ratio_min = float(valid_jnt_lr.detach().min().item()) if valid_jnt_lr.numel() > 0 else 0.0
            joint_log_ratio_max = float(valid_jnt_lr.detach().max().item()) if valid_jnt_lr.numel() > 0 else 0.0
            action_ratio_nonfinite = float((~torch.isfinite(ratio_action[valid_bool].detach())).float().sum().item())
            joint_ratio_nonfinite = float((~torch.isfinite(joint_ratio[valid_bool].detach())).float().sum().item())
            new_value_abs_max = float(new_vals[valid_bool].detach().abs().max().item()) if valid_bool.any() else 0.0
            return_abs_max = float(mb["returns"][valid_bool].detach().abs().max().item()) if valid_bool.any() else 0.0
            advantage_abs_max = float(mb["advantages"][valid_bool].detach().abs().max().item()) if valid_bool.any() else 0.0
            action_entropy_stat = masked_mean(a_entropy)
            direction_rel_frac = (direction_weight.sum() / valid_count).item()
        else:
            action_log_ratio_min = float(log_ratio_action.detach().min().item())
            action_log_ratio_max = float(log_ratio_action.detach().max().item())
            joint_log_ratio_min = float(joint_log_ratio.detach().min().item())
            joint_log_ratio_max = float(joint_log_ratio.detach().max().item())
            action_ratio_nonfinite = float((~torch.isfinite(ratio_action.detach())).float().sum().item())
            joint_ratio_nonfinite = float((~torch.isfinite(joint_ratio.detach())).float().sum().item())
            new_value_abs_max = float(new_vals.detach().abs().max().item())
            return_abs_max = float(mb["returns"].detach().abs().max().item())
            advantage_abs_max = float(mb["advantages"].detach().abs().max().item())
            action_entropy_stat = a_entropy.mean().item()
            direction_rel_frac = direction_weight.mean().item()

        relevant_dir_ratios = ratio_direction[direction_relevant]
        direction_ratio_nonfinite = float(
            (~torch.isfinite(relevant_dir_ratios.detach())).float().sum().item()
        ) if relevant_dir_ratios.numel() > 0 else 0.0
        
        aux_info = {
            "halt_bce": 0.0, "ponder": 0.0, "sel_entropy": 0.0,
            "lb_avg": 0.0, "lb_axis_avg": 0.0, "routing_aux_loss": 0.0,
            "head_decorr_penalty": 0.0, "head_collision_penalty": 0.0,
            "carry_router_overflow_penalty": 0.0,
            "joint_approx_kl": joint_approx_kl,
            "joint_clip_frac": joint_clip_frac,
            "joint_policy_loss": float(joint_policy_loss.detach().item()),
            "action_approx_kl": action_approx_kl,
            "action_clip_frac": action_clip_frac,
            "direction_approx_kl": direction_approx_kl,
            "direction_clip_frac": direction_clip_frac,
            "action_entropy": action_entropy_stat,
            "direction_entropy": direction_entropy.item(),
            "direction_relevant_fraction": direction_rel_frac,
            "action_log_ratio_min": action_log_ratio_min,
            "action_log_ratio_max": action_log_ratio_max,
            "direction_log_ratio_min": direction_log_ratio_min,
            "direction_log_ratio_max": direction_log_ratio_max,
            "joint_log_ratio_min": joint_log_ratio_min,
            "joint_log_ratio_max": joint_log_ratio_max,
            "action_ratio_nonfinite": action_ratio_nonfinite,
            "direction_ratio_nonfinite": direction_ratio_nonfinite,
            "joint_ratio_nonfinite": joint_ratio_nonfinite,
            "new_value_abs_max": new_value_abs_max,
            "return_abs_max": return_abs_max,
            "advantage_abs_max": advantage_abs_max,
        }
        


        if any(c > 0.0 for c in (self.halt_coef, self.ponder_penalty, abs(self.sel_ent_coef))):
            aux = self._aux_halting_terms(mb['state_inputs'][0], mb['state_inputs'][1])
            if aux is not None:
                # 1) BCE to encourage confident halting this segment (towards 1)
                if self.halt_coef > 0.0 and "halt_prob" in aux:
                    halt_prob = aux["halt_prob"].clamp(1e-6, 1-1e-6)  # [B,1]
                    target = torch.ones_like(halt_prob)
                    halt_bce = F.binary_cross_entropy(halt_prob, target)
                    aux_loss = aux_loss + self.halt_coef * halt_bce
                    aux_info["halt_bce"] = float(halt_bce.detach().item())

                # 2) Ponder penalty: encourage fewer segments (scaled to [0,1])
                if self.ponder_penalty > 0.0 and "segments_used" in aux:
                    max_segs = max(int(getattr(Config, "MODEL_MAX_SEGMENTS", 1)), 1)
                    used = aux["segments_used"].float()  # [B]
                    if max_segs > 1:
                        norm_used = (used - 1.0) / (max_segs - 1.0)
                    else:
                        norm_used = used * 0.0
                    ponder = norm_used.mean()
                    aux_loss = aux_loss + self.ponder_penalty * ponder
                    aux_info["ponder"] = float(ponder.detach().item())

                # 3) Selector entropy regularization over timesteps
                if self.sel_ent_coef != 0.0 and "selector_weights" in aux:
                    w = aux["selector_weights"].clamp_min(1e-12)          # [B,S]
                    sel_ent = -(w * w.log()).sum(dim=-1).mean()           # scalar
                    # Positive coef -> encourage more entropy (exploration)
                    # Negative coef -> encourage peaky selection (sharper halting)
                    aux_loss = aux_loss - self.sel_ent_coef * sel_ent
                    aux_info["sel_entropy"] = float(sel_ent.detach().item())

        # combine total loss
        self._last_aux = aux_info
        VALUE_LOSS_COEF = 0.5
        
        ENTROPY_ACTION_COEF = float(
            getattr(Config, "ACTION_ENTROPY_COEF", 0.0025)
        )
        ENTROPY_DIRECTION_COEF = float(
            getattr(Config, "DIRECTION_ENTROPY_COEF", 0.005)
        )
        if valid_mask is not None:
            action_entropy_mean = (a_entropy * valid_mask).sum() / valid_count
        else:
            action_entropy_mean = a_entropy.mean()
        entropy_bonus = (
            ENTROPY_ACTION_COEF * action_entropy_mean
            + ENTROPY_DIRECTION_COEF * direction_entropy
        )
        
        total_loss = (
            joint_policy_loss
            + VALUE_LOSS_COEF * value_loss
            - entropy_bonus
            + aux_loss
        )
        aux_info["entropy_bonus"] = float(entropy_bonus.detach().item())
        aux_info["aux_loss"] = float(aux_loss.detach().item())
        aux_info["objective_loss"] = float(total_loss.detach().item())

        return (action_loss, direction_loss, value_loss), (a_entropy, d_entropy), aux_loss, aux_info, total_loss

    def _finish_optimizer_step(self):
        if not hasattr(self.model, "agent_model"):
            return
        backbone = self.model.agent_model
        if hasattr(backbone, "retract_structured_bases_"):
            backbone.retract_structured_bases_()
        if hasattr(backbone, "project_structured_optimizer_moments_"):
            backbone.project_structured_optimizer_moments_(self.optimizer)
        if hasattr(backbone, "increment_optimizer_step"):
            backbone.increment_optimizer_step()

    def _snapshot_optimizer_boundary(self):
        """Capture exactly the state mutated by one optimizer boundary."""
        parameters = [
            (parameter, parameter.detach().clone())
            for parameter in self.model.parameters()
        ]
        return {
            "parameters": parameters,
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "model_optimizer_step": self._model_optimizer_step(),
        }

    @torch.no_grad()
    def _restore_optimizer_boundary(self, snapshot) -> None:
        for parameter, saved in snapshot["parameters"]:
            parameter.copy_(saved)
        self.optimizer.load_state_dict(snapshot["optimizer"])
        agent_model = getattr(self.model, "agent_model", None)
        step_count = getattr(agent_model, "optimizer_step_count", None)
        if step_count is not None:
            step_count.fill_(int(snapshot["model_optimizer_step"]))

    @torch.no_grad()
    def _measure_post_step_kl(self, data, minibatch_size: int) -> float:
        """Measure the true rollout-policy KL without updating router state."""
        N = data['actions'].size(0)
        total_kl = 0.0
        count = 0
        was_training = self.model.training
        self.model.eval()
        try:
            for start in range(0, N, minibatch_size):
                end = min(start + minibatch_size, N)
                real_count = end - start
                if real_count == minibatch_size:
                    mb_idx = list(range(start, end))
                else:
                    pad_count = minibatch_size - real_count
                    mb_idx = list(range(start, end)) + [0] * pad_count
                mb_inputs = (
                    data['state_inputs'][0][mb_idx],
                    data['state_inputs'][1][mb_idx],
                    data['state_inputs'][2][mb_idx],
                )
                new_a_logp, new_d_logp, _, _, _ = self.model.get_new_log_probs(
                    mb_inputs,
                    data['actions'][mb_idx],
                    data['directions'][mb_idx],
                    data.get('action_masks', None)[mb_idx]
                    if 'action_masks' in data else None,
                    data.get('direction_masks', None)[mb_idx]
                    if 'direction_masks' in data else None,
                )
                new_a_logp_real = new_a_logp[:real_count]
                new_d_logp_real = new_d_logp[:real_count]
                old_a_logp = data['old_action_log_probs'][start:end]
                old_d_logp = data['old_direction_log_probs'][start:end]
                direction_relevant = data['actions'][start:end].ne(Action.WAIT.value)
                direction_delta = torch.where(
                    direction_relevant,
                    new_d_logp_real - old_d_logp,
                    torch.zeros_like(new_d_logp_real),
                )
                log_ratio = (new_a_logp_real - old_a_logp) + direction_delta
                ratio = log_ratio.exp()
                chunk_kl = ((ratio - 1.0) - log_ratio).sum().item()
                total_kl += chunk_kl
                count += real_count
        finally:
            self.model.train(was_training)
        return total_kl / max(count, 1)

    def update(self, transitions, minibatch_size=None):
        """
        Accepts a (possibly mixed) list of transitions. Bootstrapping is handled
        inside `compute_returns_and_advantages()` using per-segment terminal/truncated
        markers and, if truncated, the provided V(s_{t+1}) at the boundary.
        """
        if minibatch_size is None:
            minibatch_size = int(Config.MINIBATCH_SIZE)
        if minibatch_size <= 0:
            raise ValueError("MINIBATCH_SIZE must be positive")
        self.transitions = transitions
        self.update_counter += 1
        returns, advantages = self.compute_returns_and_advantages()
        lr = self._lr_for_update(self.update_counter)
        for group in self.optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
    
        # Unpack (works for both 8-field and 10-field tuples)
        states, actions, directions, old_a_logp, old_d_logp, _, values, _ = zip(*[
            t[:8] for t in self.transitions
        ])
        
        view_batch = torch.stack([s[0] for s in states]).to(Config.DEVICE)
        state_batch = torch.stack([s[1] for s in states]).to(Config.DEVICE)
        pos_batch = torch.stack([s[2] for s in states]).to(Config.DEVICE)
        
        data = {
            'state_inputs': (view_batch, state_batch, pos_batch),
            'actions': torch.tensor(actions, dtype=torch.int64, device=Config.DEVICE),
            'directions': torch.tensor(directions, dtype=torch.int64, device=Config.DEVICE),
            'old_action_log_probs': torch.tensor(old_a_logp, dtype=torch.float32, device=Config.DEVICE).detach(),
            'old_direction_log_probs': torch.tensor(old_d_logp, dtype=torch.float32, device=Config.DEVICE).detach(),
            'old_values': torch.stack(values).to(Config.DEVICE).detach(),
            'returns': returns.to(Config.DEVICE),
            'advantages': advantages.to(Config.DEVICE),
        }
        
        if len(states[0]) >= 5:
            if states[0][3] is not None:
                data['action_masks'] = torch.stack([s[3] for s in states]).to(Config.DEVICE)
            if states[0][4] is not None:
                data['direction_masks'] = torch.stack([s[4] for s in states]).to(Config.DEVICE)

        # The stacked tensors now own the complete rollout. Release thousands of
        # per-transition tensor objects before the much larger backward pass.
        self.transitions = []
        transitions.clear()
        del states, actions, directions, old_a_logp, old_d_logp, values
    
        weighted_sums = {
            "action_loss": 0.0,
            "direction_loss": 0.0,
            "value_loss": 0.0,
        }
        sample_weight = 0
        aux_info_accumulator = {}
        
        # use CUDA events only if CUDA is available
        host_start = self._host_performance_snapshot()
        if torch.cuda.is_available() and Config.DEVICE.startswith("cuda"):
            start_time = torch.cuda.Event(enable_timing=True)
            end_time   = torch.cuda.Event(enable_timing=True)
            start_time.record()
        else:
            start_time = end_time = None
            t0 = time.monotonic()
    
        N = data['actions'].size(0)
        post_step_kls = []
        rejected_step_kls = []
        for ep in range(self.ppo_epochs):
            indices = torch.randperm(N, device=Config.DEVICE)
            self.optimizer.zero_grad(set_to_none=True)
            self.model.begin_router_state_observation_()
            boundary_snapshot = (
                self._snapshot_optimizer_boundary()
                if Config.USE_EARLY_STOPPING else None
            )
            try:
                for start in range(0, N, minibatch_size):
                    end = min(start + minibatch_size, N)
                    minibatch_samples = end - start
                    if minibatch_samples == minibatch_size:
                        mb_idx = indices[start:end]
                        valid_mask = torch.ones(
                            minibatch_size, dtype=torch.float32, device=Config.DEVICE
                        )
                    else:
                        pad_count = minibatch_size - minibatch_samples
                        tail_idx = indices[start:end]
                        pad_idx = indices[:1].repeat(pad_count)
                        mb_idx = torch.cat([tail_idx, pad_idx], dim=0)
                        valid_mask = torch.cat([
                            torch.ones(minibatch_samples, dtype=torch.float32, device=Config.DEVICE),
                            torch.zeros(pad_count, dtype=torch.float32, device=Config.DEVICE),
                        ], dim=0)

                    mb_inputs = (
                        data['state_inputs'][0][mb_idx],
                        data['state_inputs'][1][mb_idx],
                        data['state_inputs'][2][mb_idx]
                    )

                    # Diagnostics on the FIRST minibatch, not the last. The last
                    # minibatch is the ragged tail (end = min(start+mb, N)), and N
                    # varies per update, so pinning collect_aux to it gave the
                    # collect_aux=True graph a new batch size almost every update ->
                    # a Dynamo recompile per update (159s clean vs 512-698s with one).
                    # The first minibatch is always exactly minibatch_size, so the
                    # shape is fixed. Which minibatch supplies the diagnostic sample
                    # is arbitrary; its size is not.
                    collect_diagnostics = (
                        ep == self.ppo_epochs - 1 and start == 0
                    )
                    loss_batch = {
                        **{k: v[mb_idx] for k, v in data.items() if k != 'state_inputs'},
                        'state_inputs': mb_inputs,
                        'valid_mask': valid_mask,
                        '_collect_diagnostics': collect_diagnostics,
                        '_diagnostics_only': collect_diagnostics,
                    }
                    (a_loss, d_loss, v_loss), (a_ent, d_ent), aux_loss, aux_info, total_loss = self.compute_loss(
                        loss_batch
                    )

                    if not torch.isfinite(total_loss):
                        context = {
                            "stage": "loss",
                            "ppo_epoch": int(ep),
                            "minibatch_start": int(start),
                            "minibatch_end": int(end),
                            "aux_info": aux_info,
                            "action_loss": float(a_loss.detach().item()),
                            "direction_loss": float(d_loss.detach().item()),
                            "value_loss": float(v_loss.detach().item()),
                            "aux_loss": float(aux_loss.detach().item()),
                            "total_loss": float(total_loss.detach().item()),
                        }
                        self._save_ppo_forensics(data, context)
                        raise RuntimeError(
                            f"nonfinite PPO loss at epoch={ep} "
                            f"minibatch={start}:{end}: {context}"
                        )

                    # Every transition must contribute equally to the epoch objective.
                    # Dividing by the minibatch count overweights a short final batch.
                    # total_loss is normalized over valid transitions in this minibatch.
                    # Multiplying by (minibatch_samples / N) ensures each valid transition
                    # has exact weight 1/N across the whole epoch. Dummy padded rows
                    # have valid_mask = 0 and produce 0 loss and 0 gradient.
                    (total_loss * (float(minibatch_samples) / float(N))).backward()

                    if self.forensic_grad_checks:
                        bad, finite_norm, largest = self._gradient_forensics()
                        if bad or not math.isfinite(finite_norm):
                            context = {
                                "stage": "backward",
                                "ppo_epoch": int(ep),
                                "minibatch_start": int(start),
                                "minibatch_end": int(end),
                                "aux_info": aux_info,
                                "bad_gradients": bad,
                                "finite_gradient_norm": finite_norm,
                                "largest_finite_gradients": largest,
                            }
                            print(
                                f"[PPO NONFINITE] first bad backward: {context}",
                                flush=True,
                            )
                            self._save_ppo_forensics(data, context)
                            raise RuntimeError(
                                f"nonfinite PPO gradient at epoch={ep} "
                                f"minibatch={start}:{end}"
                            )

                    weighted_sums["action_loss"] += a_loss.item() * minibatch_samples
                    weighted_sums["direction_loss"] += d_loss.item() * minibatch_samples
                    weighted_sums["value_loss"] += v_loss.item() * minibatch_samples
                    sample_weight += minibatch_samples
                    for k, v in aux_info.items():
                        aux_info_accumulator[k] = (
                            aux_info_accumulator.get(k, 0.0)
                            + float(v) * minibatch_samples
                        )
            except Exception:
                self.model.discard_router_state_observation_()
                self.optimizer.zero_grad(set_to_none=True)
                raise

            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float(getattr(Config, "MODEL_GRAD_CLIP_NORM", 1.0)),
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                self._finish_optimizer_step()
            except Exception:
                bad, finite_norm, largest = self._gradient_forensics()
                context = {
                    "stage": "optimizer_boundary",
                    "ppo_epoch": int(ep),
                    "bad_gradients": bad,
                    "finite_gradient_norm": finite_norm,
                    "largest_finite_gradients": largest,
                    "last_aux_info": dict(self._last_aux),
                }
                print(
                    f"[PPO NONFINITE] optimizer boundary: {context}",
                    flush=True,
                )
                self._save_ppo_forensics(data, context)
                self.model.discard_router_state_observation_()
                self.optimizer.zero_grad(set_to_none=True)
                raise
            if Config.USE_EARLY_STOPPING:
                kl = self._measure_post_step_kl(data, minibatch_size)
                if not math.isfinite(kl) or kl > self.target_kl:
                    self._restore_optimizer_boundary(boundary_snapshot)
                    self.model.discard_router_state_observation_()
                    rejected_step_kls.append(float(kl))
                    ratio = (
                        float(self.target_kl) / max(float(kl), 1e-12)
                        if math.isfinite(kl) else 0.25
                    )
                    shrink = max(0.25, min(0.8, math.sqrt(max(0.0, ratio))))
                    for group in self.optimizer.param_groups:
                        group["lr_scale"] = (
                            float(group.get("lr_scale", 1.0)) * shrink
                        )
                    print(
                        f"[PPO KL ROLLBACK] rejected epoch {ep+1}: "
                        f"KL={kl:.6f} > {self.target_kl:.6f}; "
                        f"next LR scale x{shrink:.4f}",
                        flush=True,
                    )
                    break
                post_step_kls.append(float(kl))
            self.model.commit_router_state_observation_()
    
        if start_time is not None:
            end_time.record()
            torch.cuda.synchronize()
            compute_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            compute_time = time.monotonic() - t0
        host_end = self._host_performance_snapshot()
        process_seconds = max(
            0.0,
            host_end["process_seconds"] - host_start["process_seconds"],
        )
        throttle_start = host_start["package_throttle_ms"]
        throttle_end = host_end["package_throttle_ms"]
        throttle_delta_ms = (
            max(0.0, throttle_end - throttle_start)
            if math.isfinite(throttle_start) and math.isfinite(throttle_end)
            else float("nan")
        )
    
        torch.cuda.empty_cache()
        self.transitions = []
        self.model.release_last_forward_graph()
        gc_interval = max(0, int(getattr(Config, "GC_INTERVAL_UPDATES", 0)))
        if gc_interval and self.update_counter % gc_interval == 0:
            gc.collect()
    
        out_stats = {
            key: value / float(max(1, sample_weight))
            for key, value in weighted_sums.items()
        }
        out_stats.update({
            "compute_time": compute_time,
            "loss_transition_count": int(N),
            "loss_evaluation_count": int(sample_weight),
            "loss_ppo_epochs": int(self.ppo_epochs),
            "accepted_post_step_kl": (
                post_step_kls[-1] if post_step_kls else 0.0
            ),
            "rejected_post_step_kl": (
                rejected_step_kls[-1] if rejected_step_kls else 0.0
            ),
            "host_process_seconds": process_seconds,
            "host_parallelism": process_seconds / max(compute_time, 1e-9),
            "host_package_throttle_ms": throttle_delta_ms,
            "host_temperature_c": host_end["temperature_c"],
            "host_frequency_mhz": host_end["frequency_mhz"],
        })
        for k, weighted_sum in aux_info_accumulator.items():
            out_stats[k] = weighted_sum / float(max(1, sample_weight))
            
        return out_stats

    def check_early_stopping(self, old_logp, new_logp, epoch):
        if Config.USE_EARLY_STOPPING:
            with torch.no_grad():
                kl = (new_logp - old_logp).mean()
                if kl > self.target_kl:
                    print(f"Early stopping at epoch {epoch+1}, KL: {kl:.6f}")
                    return True
        return False

    def clear_memory(self):
        self.transitions = []

