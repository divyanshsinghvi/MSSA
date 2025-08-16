from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from ogm import OccupancyGridMap, EdgeAction, face_adjacent

class OGMEnvSeq(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        dims: Tuple[int,int,int],
        init_modules: List[Tuple[int,int,int]],
        goal_modules: List[Tuple[int,int,int]],
        obstacles: Optional[List[Tuple[int,int,int]]] = None,
        max_micro_steps: int = 3000,
        max_local_actions: int = 64,
        success_iou: float = 0.99,
        step_cost: float = 0.01,
        invalid_cost: float = 0.1,
        energy_cost: float = 0.0,
        sweep_steps: int = 12,
        sweep_points_per_axis: int = 4,
        enforce_connectivity: bool = True,
        shape_weight: float = 0.0,
        noop_penalty_scale: float = 1.0,
        noop_penalty_isolated_scale: float = 0.5,
        seed: Optional[int] = None,
        allow_180: bool = True,
    ):
        super().__init__()
        self.dims = tuple(map(int, dims))
        self.grid = OccupancyGridMap(self.dims, init_modules, obstacles or [])
        # IDs
        self._next_id = 0
        self.id2coord: Dict[int, Tuple[int,int,int]] = {}
        for c in sorted(set(map(tuple, init_modules))):
            self.id2coord[self._next_id] = c
            self._next_id += 1
        self.id2orig = dict(self.id2coord)

        self.goal_modules = set(map(tuple, goal_modules))

        self.max_micro_steps = int(max_micro_steps)
        self.max_local_actions = int(max_local_actions)
        self.success_iou = float(success_iou)
        self.step_cost = float(step_cost)
        self.invalid_cost = float(invalid_cost)
        self.energy_cost = float(energy_cost)
        self.sweep_steps = int(sweep_steps)
        self.sweep_points_per_axis = int(sweep_points_per_axis)
        self.enforce_connectivity = bool(enforce_connectivity)
        self.shape_weight = float(shape_weight)
        self.noop_penalty_scale = float(noop_penalty_scale)
        self.noop_penalty_isolated_scale = float(noop_penalty_isolated_scale)

        X,Y,Z = self.dims
        self.observation_space = spaces.Dict({
            "occ": spaces.Box(low=0, high=1, shape=(X,Y,Z), dtype=np.uint8),
            "goal": spaces.Box(low=0, high=1, shape=(X,Y,Z), dtype=np.uint8),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.max_local_actions,), dtype=np.uint8),
            "cur_coord": spaces.Box(low=0, high=max(X,Y,Z), shape=(3,), dtype=np.int32),
            "cur_index": spaces.Box(low=0, high=10_000, shape=(1,), dtype=np.int32),
            "num_modules": spaces.Box(low=0, high=10_000, shape=(1,), dtype=np.int32),
        })
        self.action_space = spaces.Discrete(self.max_local_actions)  # 0=NoOp, 1..k local moves

        self._rng = np.random.default_rng(seed)
        self._micro_step_count = 0
        self._cur_order: List[int] = []
        self._cur_idx = 0
        self._local_actions: List[Optional[EdgeAction]] = []
        self._last_iou = self._cur_iou()
        self._last_dist = 0.0
        self._allowed_nonnoop_sum = 0
        self._allowed_nonnoop_steps = 0
        self._pass_nonnoop_taken = False
        self._pass_iou_improved = False
        self._last_rejections: List[Dict[str, Any]] = []
        self._last_rej_counts: Dict[str, int] = {}
        self.allow_180 = bool(allow_180)

    # ----- helpers -----
    def _cur_iou(self) -> float:
        return OccupancyGridMap.iou(self.grid.modules, self.goal_modules)

    def _assignment_distance(self) -> float:
        cur = sorted(self.grid.modules)
        goal = sorted(self.goal_modules)
        if not cur or not goal: return 0.0
        C = np.zeros((len(cur), len(goal)), dtype=np.int32)
        for i,a in enumerate(cur):
            for j,b in enumerate(goal):
                C[i,j] = abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])
        try:
            from scipy.optimize import linear_sum_assignment
            r, c = linear_sum_assignment(C)
            return float(C[r,c].sum())
        except Exception:
            total = 0
            cur_left = set(cur); goal_left = set(goal)
            while cur_left and goal_left:
                best_d = 10**9; best_pair = None
                for a in cur_left:
                    for b in goal_left:
                        d = abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])
                        if d < best_d: best_d = d; best_pair = (a,b)
                a,b = best_pair
                total += best_d
                cur_left.remove(a); goal_left.remove(b)
            return float(total)

    def _start_new_pass(self):
        self._cur_order = sorted(self.id2coord.keys())
        self._cur_idx = 0

    def _current_id(self) -> Optional[int]:
        if self._cur_idx >= len(self._cur_order): return None
        return self._cur_order[self._cur_idx]

    def _enumerate_local_actions(self) -> np.ndarray:
        cur_id = self._current_id()
        mask = np.zeros((self.max_local_actions,), dtype=np.uint8)
        acts: List[Optional[EdgeAction]] = [None] * self.max_local_actions  # 0=NoOp
        if cur_id is None:
            self._local_actions = acts; return mask
        m = self.id2coord[cur_id]
        local, rej = self.grid.enumerate_edge_actions_for_with_reasons(
            m, self.sweep_steps, self.sweep_points_per_axis, self.enforce_connectivity,
            allow_180=self.allow_180
        )
        # store rejections
        self._last_rejections = rej
        rc: Dict[str,int] = {}
        for r in rej: rc[r['reason']] = rc.get(r['reason'], 0) + 1
        self._last_rej_counts = rc
        k = min(len(local), self.max_local_actions-1)
        for i in range(1, 1+k):
            acts[i] = local[i-1]; mask[i] = 1
        mask[0] = 1
        self._local_actions = acts
        return mask

    def _apply_local_action(self, a: int):
        cur_id = self._current_id()
        if cur_id is None: return
        if a <= 0 or a >= len(self._local_actions) or self._local_actions[a] is None:
            return
        act = self._local_actions[a]
        self.grid.apply_action(act)
        self.id2coord[cur_id] = act.end_min_corner

    # ----- gym api -----
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str,Any]] = None):
        if seed is not None: self._rng = np.random.default_rng(seed)
        if options is not None:
            init = options.get("init_modules"); goal = options.get("goal_modules"); obstacles = options.get("obstacles")
            if init is not None:
                self.grid = OccupancyGridMap(self.dims, init, obstacles or [])
                self.id2coord.clear(); self._next_id = 0
                for c in sorted(set(map(tuple, init))):
                    self.id2coord[self._next_id] = c; self._next_id += 1
                self.id2orig = dict(self.id2coord)
            if goal is not None: self.goal_modules = set(map(tuple, goal))
        self._micro_step_count = 0
        self._last_iou = self._cur_iou()
        self._last_dist = self._assignment_distance()
        self._allowed_nonnoop_sum = 0; self._allowed_nonnoop_steps = 0
        self._pass_nonnoop_taken = False; self._pass_iou_improved = False
        self._last_rejections = []; self._last_rej_counts = {}
        self._start_new_pass()
        mask = self._enumerate_local_actions()
        cur_id = self._current_id()
        cur_coord = np.array(self.id2coord[cur_id], dtype=np.int32) if cur_id is not None else np.array([0,0,0], dtype=np.int32)
        nonnoop_count = int(sum(1 for a in self._local_actions[1:] if a is not None))
        self._allowed_nonnoop_sum += nonnoop_count; self._allowed_nonnoop_steps += 1
        obs = {
            "occ": self.grid.occupancy_grid(),
            "goal": self.grid.goal_grid(self.goal_modules),
            "action_mask": mask,
            "cur_coord": cur_coord,
            "cur_index": np.array([self._cur_idx], dtype=np.int32),
            "num_modules": np.array([len(self.id2coord)], dtype=np.int32),
        }
        info = {
            "local_valid_actions": int(mask.sum()),
            "nonnoop_count": nonnoop_count,
            "allowed_nonnoop_mean": (self._allowed_nonnoop_sum / max(1, self._allowed_nonnoop_steps)),
            "iou": self._last_iou,
            "module_id": int(cur_id) if cur_id is not None else -1,
            "cur_index": int(self._cur_idx),
            "num_modules": int(len(self.id2coord)),
            "rej_counts": dict(self._last_rej_counts),
            "at_original": True,
            "neighbor_fine": (len(self.grid.neighbors_face(tuple(cur_coord)))>0),
            "nonnoop_taken": False,
        }
        return obs, info

    def step(self, action: int):
        self._micro_step_count += 1
        reward = -self.step_cost
        terminated = False
        truncated = False

        before_iou = self._last_iou

        # Identify current module before applying
        cur_id = self._current_id()
        cur_coord_before = self.id2coord[cur_id] if cur_id is not None else None
        at_original = (cur_coord_before is not None and cur_id in self.id2orig and cur_coord_before == self.id2orig[cur_id])
        neighbor_fine = False
        if cur_coord_before is not None:
            neighs = [n for n in self.grid.neighbors_face(cur_coord_before) if n != cur_coord_before]
            neighbor_fine = (len(neighs) > 0)

        # Apply
        self._apply_local_action(int(action))
        iou = self._cur_iou()
        delta_iou = (iou - before_iou)
        reward += delta_iou
        self._last_iou = iou

        # NoOp nuanced penalties
        nonnoop_available = any(a is not None for a in self._local_actions[1:])
        nonnoop_taken = (action > 0 and action < len(self._local_actions) and self._local_actions[action] is not None)
        if nonnoop_available and not nonnoop_taken:
            if not at_original:
                reward -= self.noop_penalty_scale * self.step_cost
            else:
                if not neighbor_fine:
                    reward -= self.noop_penalty_isolated_scale * self.step_cost

        if nonnoop_taken: self._pass_nonnoop_taken = True
        if delta_iou > 0: self._pass_iou_improved = True

        # Distance shaping
        if self.shape_weight != 0.0:
            cur_dist = self._assignment_distance()
            reward += self.shape_weight * (self._last_dist - cur_dist)
            self._last_dist = cur_dist

        # Advance to next module
        self._cur_idx += 1
        if self._cur_idx >= len(self._cur_order):
            # Early truncate if no progress in the whole pass
            if not self._pass_nonnoop_taken and not self._pass_iou_improved:
                truncated = True
            self._start_new_pass()
            self._pass_nonnoop_taken = False; self._pass_iou_improved = False

        # Success / horizon
        if iou >= self.success_iou:
            reward += 10.0; terminated = True
        if self._micro_step_count >= self.max_micro_steps:
            truncated = True

        # Next obs + mask
        mask = self._enumerate_local_actions()
        cur_id2 = self._current_id()
        cur_coord = np.array(self.id2coord[cur_id2], dtype=np.int32) if cur_id2 is not None else np.array([0,0,0], dtype=np.int32)

        nonnoop_count = int(sum(1 for a in self._local_actions[1:] if a is not None))
        self._allowed_nonnoop_sum += nonnoop_count; self._allowed_nonnoop_steps += 1

        obs = {
            "occ": self.grid.occupancy_grid(),
            "goal": self.grid.goal_grid(self.goal_modules),
            "action_mask": mask,
            "cur_coord": cur_coord,
            "cur_index": np.array([self._cur_idx], dtype=np.int32),
            "num_modules": np.array([len(self.id2coord)], dtype=np.int32),
        }
        info = {
            "local_valid_actions": int(mask.sum()),
            "nonnoop_count": nonnoop_count,
            "allowed_nonnoop_mean": (self._allowed_nonnoop_sum / max(1, self._allowed_nonnoop_steps)),
            "iou": iou,
            "module_id": int(cur_id2) if cur_id2 is not None else -1,
            "cur_index": int(self._cur_idx),
            "num_modules": int(len(self.id2coord)),
            "rej_counts": dict(self._last_rej_counts),
            "at_original": bool(at_original),
            "neighbor_fine": bool(neighbor_fine),
            "nonnoop_taken": bool(nonnoop_taken),
            "allow_180" : int(getattr(self, "allow_180", False)),
        }
        return obs, float(reward), terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        return self._enumerate_local_actions().astype(bool)

    def get_last_rejections(self):
        return list(self._last_rejections)

    def render(self):
        print("Modules:", sorted(self.grid.modules), "IoU:", self._cur_iou(), "idx:", self._cur_idx)

    def close(self):
        pass
