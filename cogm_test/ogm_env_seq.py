#ogm_env_seq.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ogm import OccupancyGridMap, EdgeAction, face_adjacent

def _bbox(points: List[Tuple[int,int,int]]):
    xs = [p[0] for p in points]; ys = [p[1] for p in points]; zs = [p[2] for p in points]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

def _compute_obs_origin(mods: List[Tuple[int,int,int]],
                        goals: List[Tuple[int,int,int]],
                        window: Tuple[int,int,int], pad: int = 2,
                        center_mode: str = "bbox_min") -> Tuple[int,int,int]:
    pts = (list(mods) or []) + (list(goals) or [])
    if not pts:
        return (0,0,0)
    (mnx,mny,mnz), (mxx,mxy,mxz) = _bbox(pts)
    if center_mode == "bbox_center":
        cx = (mnx + mxx) // 2
        cy = (mny + mxy) // 2
        cz = (mnz + mxz) // 2
        ox = cx - window[0]//2
        oy = cy - window[1]//2
        oz = cz - window[2]//2
    else:
        ox = mnx - pad; oy = mny - pad; oz = mnz - pad
    return (ox,oy,oz)

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
        step_cost: float = 0.001,
        invalid_cost: float = 0.1,
        energy_cost: float = 0.0,
        sweep_steps: int = 12,
        sweep_points_per_axis: int = 4,
        enforce_connectivity: bool = True,
        shape_weight: float = 0.004,
        noop_penalty_scale: float = 1.0,
        noop_penalty_isolated_scale: float = 0.5,
        noop_tie_break_penalty: float = 0.001,
        seed: Optional[int] = None,
        allow_180: bool = True,
        patience_no_progress: int = 10,
        obs_mode: str = "voxel",                 # "voxel" | "graph" | "both"
        obs_window: Tuple[int,int,int] = (12,12,6),
        obs_pad: int = 2,
        obs_center_mode: str = "bbox_min",       # "bbox_min" | "bbox_center"
        max_nodes: int = 64,                     # for graph mode padding
        randomize_order: bool = True,
        loop_penalty: float = 0.05,
        loop_window: int = 5000,
        loop_hash_size: int = 20000,
        trans_weight: float = 0.05,
        infinite_world: bool = False,
        boundary_center: Tuple[int,int,int] = (0,0,0),
        boundary_l1_radius: float = 0.0,
        boundary_penalty: float = 0.0,
        improve_bonus:  float = 0.3,
        goal_snap_bonus: float = 5.0,
        neighbor_match_bonus: float = 2.0,
        assign_mode: str = "sticky_pass",   # "dynamic" | "sticky_pass" | "sticky_episode"
        order_mode:  str = "round_robin",   # "round_robin" | "largest_first"
        graph_mode: str = "combined",        # "separate" (old) | "combined"
        goal_edge_mode: str = "radius",      # "radius" | "knn"
        goal_edge_radius: int = 2,           # L1 radius for module↔goal edges (if radius mode)
        goal_edge_k: int = 2,                # k nearest goals per module (if knn mode)
        connect_goal_goal: bool = False,     # usually False; can turn on to wire goal-goal proximity
        boundary_cap: float = 100.0,
        boundary_norm_by_modules: bool = True,
    ):
        super().__init__()
        self.dims = tuple(map(int, dims))
        self.grid = OccupancyGridMap(self.dims, init_modules, obstacles or [], infinite_world=infinite_world)
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
        self.noop_tie_break_penalty = float(noop_tie_break_penalty)

        self.obs_mode = str(obs_mode)
        self.obs_window = tuple(map(int, obs_window))
        self.obs_pad = int(obs_pad)
        self.obs_center_mode = str(obs_center_mode)
        self.max_nodes = int(max_nodes)
        self.randomize_order = bool(randomize_order)
        self.loop_penalty = float(loop_penalty)
        self.loop_hash_size = int(loop_hash_size)
        self.trans_weight = float(trans_weight)
        self._visited = set()
        self._visit_q = deque(maxlen=int(loop_window))
        self._last_cdist = 0.0


        # RL bookkeeping
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
        self.patience_no_progress = int(patience_no_progress)
        self._stall_passes = 0
        self.infinite_world = bool(infinite_world)                    
        self.boundary_center = tuple(int(v) for v in boundary_center) 
        self.boundary_l1_radius = float(boundary_l1_radius)           
        self.boundary_penalty = float(boundary_penalty)
        self.improve_bonus = float(improve_bonus)   
        self.goal_snap_bonus = float(goal_snap_bonus)
        self.neighbor_match_bonus = float(neighbor_match_bonus)
        self._snapped_ids: set[int] = set()   # award snap bonus once per module
        # ----- observation space -----
        Xw,Yw,Zw = self.obs_window
        base_spaces = {
            "action_mask": spaces.Box(low=0, high=1, shape=(self.max_local_actions,), dtype=np.uint8),
            "cur_coord": spaces.Box(low=-10_000, high=10_000, shape=(3,), dtype=np.int32),
            "cur_index": spaces.Box(low=0, high=10_000, shape=(1,), dtype=np.int32),
            "num_modules": spaces.Box(low=0, high=10_000, shape=(1,), dtype=np.int32),
            "obs_origin":             spaces.Box(low=-10_000, high=10_000, shape=(3,), dtype=np.int32),
            "centroid":               spaces.Box(low=-1e4,   high=1e4,     shape=(3,), dtype=np.float32),
            "goal_centroid":          spaces.Box(low=-1e4,   high=1e4,     shape=(3,), dtype=np.float32),
            "centroid_to_goal":       spaces.Box(low=-1e4,   high=1e4,     shape=(3,), dtype=np.float32),
            "centroid_to_boundary":   spaces.Box(low=-1e4,   high=1e4,     shape=(3,), dtype=np.float32),
            "boundary_overage_obs":   spaces.Box(low=0.0,    high=1e4,     shape=(1,), dtype=np.float32),
        }

        if self.obs_mode in ("voxel", "both"):
            base_spaces.update({
                "occ": spaces.Box(low=0, high=1, shape=(Xw,Yw,Zw), dtype=np.uint8),
                "goal": spaces.Box(low=0, high=1, shape=(Xw,Yw,Zw), dtype=np.uint8),
            })
        if self.obs_mode in ("graph", "both"):
            base_spaces.update({
                "node_coords": spaces.Box(low=-10_000, high=10_000, shape=(self.max_nodes,3), dtype=np.int32),
                "node_mask": spaces.Box(low=0, high=1, shape=(self.max_nodes,), dtype=np.uint8),
                "adj": spaces.Box(low=0, high=1, shape=(self.max_nodes,self.max_nodes), dtype=np.uint8),
                # Optional: goals as a set (pad/truncate to max_nodes as well)
                "goal_coords": spaces.Box(low=-10_000, high=10_000, shape=(self.max_nodes,3), dtype=np.int32),
                "goal_mask": spaces.Box(low=0, high=1, shape=(self.max_nodes,), dtype=np.uint8),

                "combined_coords": spaces.Box(low=-10_000, high=10_000, shape=(self.max_nodes,3), dtype=np.int32),
                "combined_mask":   spaces.Box(low=0, high=1,       shape=(self.max_nodes,),   dtype=np.uint8),
                "combined_type":   spaces.Box(low=0, high=1,       shape=(self.max_nodes,),   dtype=np.uint8),  # 0=module, 1=goal
                "combined_adj":    spaces.Box(low=0, high=1,       shape=(self.max_nodes,self.max_nodes), dtype=np.uint8),
            
            })

        self.observation_space = spaces.Dict(base_spaces)
        self.action_space = spaces.Discrete(self.max_local_actions)  # 0=NoOp, 1..k local moves
        self.assign_mode = str(assign_mode)
        self.order_mode  = str(order_mode)
        self._id2goal = {}      # module_id -> goal cell (sticky)
        self.graph_mode = str(graph_mode)
        self.goal_edge_mode = str(goal_edge_mode)
        self.goal_edge_radius = int(goal_edge_radius)
        self.goal_edge_k = int(goal_edge_k)
        self.connect_goal_goal = bool(connect_goal_goal)
        self.boundary_cap = float(boundary_cap)
        self.boundary_norm_by_modules = bool(boundary_norm_by_modules)

    def _goal_neighbors(self, cell: Tuple[int,int,int]) -> set[Tuple[int,int,int]]:
        """Face-adjacent goal cells around `cell` (absolute coords)."""
        if cell is None:
            return set()
        x,y,z = cell
        cand = [(x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)]
        G = self.goal_modules
        return {c for c in cand if c in G}


    def _build_obs_graph_combined(self, origin: Tuple[int,int,int]) -> Dict[str, Any]:
        K = self.max_nodes

        # 1) Gather nodes (modules then goals), shifted by origin
        mod_ids = sorted(self.id2coord.keys())
        mods_abs = [self.id2coord[i] for i in mod_ids]
        goals_abs = sorted(self.goal_modules)

        nodes_abs = mods_abs + goals_abs
        types = [0]*len(mods_abs) + [1]*len(goals_abs)  # 0=module, 1=goal

        rel = [(x-origin[0], y-origin[1], z-origin[2]) for (x,y,z) in nodes_abs]

        N = min(len(rel), K)
        coords = np.zeros((K,3), dtype=np.int32)
        mask   = np.zeros((K,),  dtype=np.uint8)
        ntype  = np.zeros((K,),  dtype=np.uint8)
        for i in range(N):
            coords[i] = rel[i]
            mask[i] = 1
            ntype[i] = types[i]

        # 2) Build adjacency
        A = np.zeros((K,K), dtype=np.uint8)

        # indices in the combined array
        n_mod  = min(len(mods_abs), K)
        n_goal = max(0, min(len(goals_abs), K - n_mod))

        # (a) module-module: face adjacency in absolute coords
        for i in range(n_mod):
            a = mods_abs[i]
            for j in range(i+1, n_mod):
                b = mods_abs[j]
                if face_adjacent(a,b):
                    A[i,j] = A[j,i] = 1

        # (b) module-goal edges by radius or kNN (using L1 in absolute coords)
        def l1(p,q): return abs(p[0]-q[0]) + abs(p[1]-q[1]) + abs(p[2]-q[2])
        if n_goal > 0:
            goals_abs_trunc = goals_abs[:n_goal]
            for i in range(n_mod):
                a = mods_abs[i]
                if self.goal_edge_mode == "radius":
                    for jg, g in enumerate(goals_abs_trunc):
                        if l1(a,g) <= self.goal_edge_radius:
                            j = n_mod + jg
                            if j < K: A[i,j] = A[j,i] = 1
                else:  # knn
                    dists = [(jg, l1(a,g)) for jg,g in enumerate(goals_abs_trunc)]
                    dists.sort(key=lambda t: t[1])
                    for jg, _ in dists[:self.goal_edge_k]:
                        j = n_mod + jg
                        if j < K: A[i,j] = A[j,i] = 1

        # (c) optional goal-goal connectivity (e.g. grid neighbors in goal shape)
        if self.connect_goal_goal and n_goal > 0:
            for jg in range(n_goal):
                ga = goals_abs[jg]
                for kg in range(jg+1, n_goal):
                    gb = goals_abs[kg]
                    if face_adjacent(ga, gb):
                        ia = n_mod + jg; ib = n_mod + kg
                        if ia < K and ib < K:
                            A[ia, ib] = A[ib, ia] = 1

        result = {
            "combined_coords": coords,
            "combined_mask":   mask,
            "combined_type":   ntype,
            "combined_adj":    A,
        }
        diags = {
            "num_graph_nodes": int(mask.sum()),
            "num_graph_edges": int(A.sum()),   # undirected counted twice
            "num_goal_nodes":  int(n_goal),
        }
        return result, diags


    def _compute_sticky_assignment(self):
        # Map each module id to one goal cell (1-1), using Hungarian once.
        ids = sorted(self.id2coord.keys())
        mods = [self.id2coord[i] for i in ids]
        goals = sorted(self.goal_modules)
        if not ids or not goals: 
            self._id2goal = {}
            return
        C = np.zeros((len(mods), len(goals)), dtype=np.int32)
        for i, a in enumerate(mods):
            for j, b in enumerate(goals):
                C[i, j] = abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])
        try:
            from scipy.optimize import linear_sum_assignment
            r, c = linear_sum_assignment(C)
        except Exception:
            # greedy fallback
            r, c = [], []
            taken = set()
            for i, a in enumerate(mods):
                best, bj = 10**9, None
                for j, b in enumerate(goals):
                    if j in taken: continue
                    d = abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])
                    if d < best: best, bj = d, j
                taken.add(bj); r.append(i); c.append(bj)
        self._id2goal = { ids[i]: goals[j] for i, j in zip(r, c) }

    # ----- helpers -----
    def _centroid_distance(self) -> float:
        if not self.grid.modules or not self.goal_modules:
            return 0.0
        mx = np.mean([x for x,_,_ in self.grid.modules])
        my = np.mean([y for _,y,_ in self.grid.modules])
        mz = np.mean([z for *_,z in self.grid.modules])
        gx = np.mean([x for x,_,_ in self.goal_modules])
        gy = np.mean([y for _,y,_ in self.goal_modules])
        gz = np.mean([z for *_,z in self.goal_modules])
        return abs(mx-gx) + abs(my-gy) + abs(mz-gz)

    def _state_fingerprint(self) -> int:
        # order-invariant hash of the assembly (exact coordinates)
        return hash(tuple(sorted(self.grid.modules)))

    # def _start_new_pass(self):
    #     order = sorted(self.id2coord.keys())
    #     if self.randomize_order:
    #         self._rng.shuffle(order)
    #     self._cur_order = order
    #     self._cur_idx = 0

    def _cur_iou(self) -> float:
        return OccupancyGridMap.iou(self.grid.modules, self.goal_modules)

    def _assembly_centroid(self) -> Tuple[float,float,float]:
        if not self.grid.modules: return (0.0,0.0,0.0)
        xs, ys, zs = zip(*self.grid.modules)
        return (float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs)))

    def _boundary_overage(self) -> float:
        if self.boundary_penalty <= 0.0 or self.boundary_l1_radius <= 0.0:
            return 0.0
        cx,cy,cz = self._assembly_centroid()
        bx,by,bz = self.boundary_center
        l1 = abs(cx-bx) + abs(cy-by) + abs(cz-bz)
        return max(0.0, l1 - self.boundary_l1_radius)

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
        try:
            local, rej = self.grid.enumerate_edge_actions_for_with_reasons(
                m, self.sweep_steps, self.sweep_points_per_axis, self.enforce_connectivity,
                allow_180=self.allow_180
            )
        except TypeError:
            local, rej = self.grid.enumerate_edge_actions_for_with_reasons(
                m, self.sweep_steps, self.sweep_points_per_axis, self.enforce_connectivity
            )
        self._last_rejections = rej
        self._last_rej_counts = {}
        for r in rej:
            k = r.get("reason","unknown")
            self._last_rej_counts[k] = self._last_rej_counts.get(k, 0) + 1
        # ALSO (optional):
        self._last_rej_by_module = {str(self.id2coord[cur_id]): dict(self._last_rej_counts)}
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

    # ----- obs builders -----
    def _build_obs_voxel(self, origin: Tuple[int,int,int]) -> Dict[str, Any]:
        Xw,Yw,Zw = self.obs_window
        return {
            "occ": self.grid.occupancy_window(origin, self.obs_window),
            "goal": self.grid.goal_window(self.goal_modules, origin, self.obs_window),
            "obs_origin": np.array(origin, dtype=np.int32),
        }

    def _start_new_pass(self):
        ids = sorted(self.id2coord.keys())
        if self.order_mode == "largest_first" and self._id2goal:
            def d(i):
                a = self.id2coord[i]; b = self._id2goal.get(i, a)
                return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])
            ids.sort(key=d, reverse=True)
        else:
            if self.randomize_order:
                self._rng.shuffle(ids)
        # --- NEW: movable-first (descending by number of valid actions) ---
        # compute counts once; small N so cheap
        counts = {i: self._count_valid_actions_for(i) for i in ids}
        # stable sort: modules with >0 moves first, then by count desc
        ids.sort(key=lambda i: (counts[i] == 0, -counts[i]))

        self._cur_order = ids
        self._cur_idx = 0
        if self.assign_mode == "sticky_pass":
            self._compute_sticky_assignment()

        # start the pass at a module that actually has moves if possible
        self._advance_to_movable()


    def _build_obs_graph(self, origin: Tuple[int,int,int]) -> Dict[str, Any]:
        # nodes (relative to origin to keep values small)
        nodes = [self.id2coord[i] for i in sorted(self.id2coord.keys())]
        rel_nodes = [(x-origin[0], y-origin[1], z-origin[2]) for (x,y,z) in nodes]
        K = self.max_nodes
        node_coords = np.full((K,3), fill_value=0, dtype=np.int32)
        node_mask = np.zeros((K,), dtype=np.uint8)
        for i,(x,y,z) in enumerate(rel_nodes[:K]):
            node_coords[i] = (x,y,z); node_mask[i] = 1
        # adjacency (face-adjacent in current assembly)
        adj = np.zeros((K,K), dtype=np.uint8)
        for i,a in enumerate(nodes[:K]):
            for j,b in enumerate(nodes[:K]):
                if i!=j and face_adjacent(a,b):
                    adj[i,j] = 1
        # goals (truncate/pad)
        goals = list(sorted(self.goal_modules))
        rel_goals = [(x-origin[0], y-origin[1], z-origin[2]) for (x,y,z) in goals]
        goal_coords = np.full((K,3), fill_value=0, dtype=np.int32)
        goal_mask = np.zeros((K,), dtype=np.uint8)
        for i,(x,y,z) in enumerate(rel_goals[:K]):
            goal_coords[i] = (x,y,z); goal_mask[i] = 1
        return {
            "node_coords": node_coords,
            "node_mask": node_mask,
            "adj": adj,
            "goal_coords": goal_coords,
            "goal_mask": goal_mask,
        }

    def _count_valid_actions_for(self, module_id: int) -> int:
        """How many non-NoOp actions are available for this module id?"""
        m = self.id2coord[module_id]
        try:
            local, _ = self.grid.enumerate_edge_actions_for_with_reasons(
                m, self.sweep_steps, self.sweep_points_per_axis, self.enforce_connectivity,
                allow_180=self.allow_180
            )
        except TypeError:
            local, _ = self.grid.enumerate_edge_actions_for_with_reasons(
                m, self.sweep_steps, self.sweep_points_per_axis, self.enforce_connectivity
            )
        return len(local)

    def _advance_to_movable(self):
        """
        Advance _cur_idx until we find a module with at least one non-noop action.
        Returns (mask, cur_id, cur_coord). Makes at most len(_cur_order) hops to avoid loops.
        """
        # If order is empty or we’re at end, start a pass now
        if self._cur_idx >= len(self._cur_order):
            self._start_new_pass()

        max_hops = max(1, len(self._cur_order))
        hops = 0
        while True:
            cur_id = self._current_id()
            if cur_id is None:
                # No modules: only NoOp available
                mask = np.zeros((self.max_local_actions,), dtype=np.uint8)
                mask[0] = 1
                return mask, None, np.array([0,0,0], dtype=np.int32)

            # Build action table for this module
            mask = self._enumerate_local_actions()
            nonnoop = int(mask[1:].sum())

            if nonnoop > 0:
                # Found a movable module
                cur_coord = np.array(self.id2coord[cur_id], dtype=np.int32)
                return mask, cur_id, cur_coord

            # No valid move for this module → hop to next
            self._cur_idx += 1
            if self._cur_idx >= len(self._cur_order):
                # end of pass, start a new one
                self._start_new_pass()

            hops += 1
            if hops >= max_hops:
                # None of the modules have a move. Fall back to NoOp on current.
                cur_id = self._current_id()
                mask = np.zeros((self.max_local_actions,), dtype=np.uint8)
                mask[0] = 1
                cur_coord = (np.array(self.id2coord[cur_id], dtype=np.int32)
                            if cur_id is not None else np.array([0,0,0], dtype=np.int32))
                return mask, cur_id, cur_coord


    def _make_obs(self, mask: np.ndarray, cur_coord: np.ndarray) -> Dict[str, Any]:
        modules = list(self.grid.modules)
        goals = list(self.goal_modules)
        origin = _compute_obs_origin(modules, goals, self.obs_window, self.obs_pad, self.obs_center_mode)

        obs: Dict[str, Any] = {
            "action_mask": mask,
            "cur_coord": cur_coord,
            "cur_index": np.array([self._cur_idx], dtype=np.int32),
            "num_modules": np.array([len(self.id2coord)], dtype=np.int32),
        }
        self._last_graph_diags = {}



        if self.obs_mode in ("voxel","both"):
            obs.update(self._build_obs_voxel(origin))
        if self.obs_mode in ("graph","both"):
            obs.update(self._build_obs_graph(origin))
            if self.graph_mode == "combined":
                comb, diags = self._build_obs_graph_combined(origin)
                obs.update(comb)
                self._last_graph_diags = diags
            
        cx,cy,cz = self._assembly_centroid()
        gx = np.mean([x for x,_,_ in self.goal_modules]) if self.goal_modules else 0.0
        gy = np.mean([y for _,y,_ in self.goal_modules]) if self.goal_modules else 0.0
        gz = np.mean([z for *_,z in self.goal_modules]) if self.goal_modules else 0.0
        bx,by,bz = self.boundary_center
        obs.update({
            "obs_origin": np.array(origin, np.int32),              # add even if graph mode
            "centroid": np.array([cx,cy,cz], np.float32),
            "goal_centroid": np.array([gx,gy,gz], np.float32),
            "centroid_to_goal": np.array([gx-cx, gy-cy, gz-cz], np.float32),
            "centroid_to_boundary": np.array([bx-cx, by-cy, bz-cz], np.float32),
            "boundary_overage_obs": np.array([self._boundary_overage()], np.float32),
        })

        return obs, origin

    # ----- gym api -----
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str,Any]] = None):



        if seed is not None: self._rng = np.random.default_rng(seed)
        if options is not None:
            init = options.get("init_modules"); goal = options.get("goal_modules"); obstacles = options.get("obstacles")
            if init is not None:
                self.grid = OccupancyGridMap(self.dims, init, obstacles or [], infinite_world=self.infinite_world)
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
        self._stall_passes = 0
        if self.assign_mode == "sticky_episode":
            self._compute_sticky_assignment()
        self._start_new_pass()
        # self._advance_to_movable()
        mask, cur_id, cur_coord = self._advance_to_movable()
        nonnoop_count = int(sum(1 for a in self._local_actions[1:] if a is not None))
        self._allowed_nonnoop_sum += nonnoop_count; self._allowed_nonnoop_steps += 1
        self._visited.clear(); self._visit_q.clear()
        self._last_cdist = self._centroid_distance()
        self._snapped_ids.clear()
        self._last_over = self._boundary_overage()
        obs, origin = self._make_obs(mask, cur_coord)
        info = {
            "local_valid_actions": int(mask.sum()),
            "valid_actions": int(mask.sum()),
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
            "allow_180": int(self.allow_180),
            "obs_origin": origin,
            "boundary_overage": float(self._boundary_overage()),
            "infinite_world": 1 if self.infinite_world else 0,
            "centroid_dist": float(self._last_cdist),
            "module_coord": tuple(int(v) for v in cur_coord.tolist()),
            "module_orig_coord": self.id2orig.get(cur_id, tuple(cur_coord.tolist())) if cur_id is not None else (-1,-1,-1),
            **getattr(self, "_last_graph_diags", {}),
        }
        return obs, info

    def step(self, action: int):
        self._micro_step_count += 1
        reward = 0
        terminated = False
        truncated = False

        before_iou = self._last_iou

        cur_id = self._current_id()
        cur_coord_before = self.id2coord[cur_id] if cur_id is not None else None
        at_original = (cur_coord_before is not None and cur_id in self.id2orig and cur_coord_before == self.id2orig[cur_id])
        neighbor_fine = False
        if cur_coord_before is not None:
            neighs = [n for n in self.grid.neighbors_face(cur_coord_before) if n != cur_coord_before]
            neighbor_fine = (len(neighs) > 0)

        cur_id = self._current_id()
        g = self._id2goal.get(cur_id) if self._id2goal else None
        d_before = (abs(cur_coord_before[0]-g[0]) + abs(cur_coord_before[1]-g[1]) + abs(cur_coord_before[2]-g[2])) if (g and cur_coord_before) else None


        self._apply_local_action(int(action))

        d_after = None
        if g is not None and cur_id is not None:
            cur_after = self.id2coord[cur_id]
            d_after = abs(cur_after[0]-g[0]) + abs(cur_after[1]-g[1]) + abs(cur_after[2]-g[2])
        r_assign_local = 0.0
        if d_before is not None and d_after is not None:
            r_assign_local = self.shape_weight * (d_before - d_after)  # positive if closer to its own goal

        # ---------- strong positive feedback when position & neighbors are correct ----------
        snap_hit = False
        r_goal_snap = 0
        r_neibour_match = 0
        neighbor_match = 0.0
        if g is not None and cur_id is not None:
            cur_after = self.id2coord[cur_id]
            if cur_after == g and cur_id not in self._snapped_ids:
                # one-time big bonus when a module lands on its own goal cell
                r_goal_snap = self.goal_snap_bonus
                snap_hit = True
                self._snapped_ids.add(cur_id)
            # If it is at the correct goal cell, reward neighbor correctness too
            if cur_after == g:
                goal_neighs = self._goal_neighbors(g)
                cur_neighs  = set(self.grid.neighbors_face(cur_after))
                # score = fraction of expected goal neighbors already present
                denom = max(1, len(goal_neighs))
                neighbor_match = len(goal_neighs & cur_neighs) / denom
                r_neibour_match += self.neighbor_match_bonus * neighbor_match


        iou = self._cur_iou()
        self._last_iou = iou

        nonnoop_available = any(a is not None for a in self._local_actions[1:])
        nonnoop_taken = (action > 0 and action < len(self._local_actions) and self._local_actions[action] is not None)

        # assignment shaping
        delta_assign = 0.0; r_assign = 0.0
        if self.shape_weight != 0.0:
            cur_dist = self._assignment_distance()
            delta_assign = (self._last_dist - cur_dist)         # >0 is good
            r_assign = self.shape_weight * delta_assign
            self._last_dist = cur_dist



        # centroid shaping
        delta_centroid = 0.0; r_centroid = 0.0
        if self.trans_weight != 0.0:
            cur_cdist = self._centroid_distance()
            delta_centroid = (self._last_cdist - cur_cdist)     # >0 is good
            r_centroid = self.trans_weight * delta_centroid
            self._last_cdist = cur_cdist

        has_improving = False
        if cur_id is not None and g is not None:
            for act in self._local_actions[1:]:
                if act is None: continue
                end = act.end_min_corner
                da = d_before - (abs(end[0]-g[0]) + abs(end[1]-g[1]) + abs(end[2]-g[2]))
                if da > 0:
                    has_improving = True; break

        # has_improving = (delta_assign > 0) or (delta_centroid > 0)
        r_noop = 0.0
        if nonnoop_available and not nonnoop_taken and has_improving:
            r_noop = - max(3.0 * self.step_cost, 0.005)

        r_base = -self.step_cost
        r_iou  = (iou - before_iou)

        if nonnoop_taken: self._pass_nonnoop_taken = True
        if r_iou > 0: self._pass_iou_improved = True






        improve_iou = (r_iou > 0.0)
        improve_assign = (delta_assign > 0.0)
        improve_centroid = (delta_centroid > 0.0)
        r_improve = self.improve_bonus if (improve_iou or improve_assign or improve_centroid) else 0.0
        # reward += r_improve

        # --- loop penalty: discourage revisiting exact same assembly state ---
        loop_hit = False
        fp = self._state_fingerprint()
        if fp in self._visited:
            loop_hit = True
            # reward -= self.loop_penalty
        else:
            self._visited.add(fp)
            self._visit_q.append(fp)
            # bound memory
            if len(self._visited) > self.loop_hash_size and len(self._visit_q) > 0:
                old = self._visit_q.popleft()
                if old in self._visited:
                    self._visited.discard(old)

        r_loop = 0
        if self._micro_step_count > 500 and loop_hit:
            r_loop = -self.loop_penalty

        
        over_now = self._boundary_overage()
        over = (over_now - self._last_over)
        if self.boundary_norm_by_modules and len(self.id2coord) > 0:
            over = over / float(len(self.id2coord))
        over = min(over, self.boundary_cap)
        r_bound = - self.boundary_penalty * over
        # r_bound = max(min(r_bound, +0.1), -0.1)
        self._last_over = over_now
        # over = max(0.0, cur_cdist - self._last_cdist)
        # reward -= 0.05 * over

        reward += r_base + r_iou + r_assign + r_centroid + r_noop + r_loop + r_bound + r_improve + 1.5 * r_assign_local + r_neibour_match + r_goal_snap
        reward = np.clip(reward, -5.0, 5.0)

        self._cur_idx += 1
        if self._cur_idx >= len(self._cur_order):
            if not self._pass_nonnoop_taken and not self._pass_iou_improved:
                self._stall_passes += 1
            else:
                self._stall_passes = 0
            if self._stall_passes >= self.patience_no_progress:
                truncated = True
            self._start_new_pass()
            self._pass_nonnoop_taken = False; self._pass_iou_improved = False

        self._advance_to_movable()

        if iou >= self.success_iou:
            reward += 10.0; terminated = True
        if self._micro_step_count >= self.max_micro_steps:
            truncated = True

        mask, cur_id2, cur_coord = self._advance_to_movable()


        nonnoop_count = int(sum(1 for a in self._local_actions[1:] if a is not None))
        self._allowed_nonnoop_sum += nonnoop_count; self._allowed_nonnoop_steps += 1

        nonnoop_avail = int(mask[1:].sum())

        obs, origin = self._make_obs(mask, cur_coord)



        info = {
            "local_valid_actions": int(mask.sum()),
            "valid_actions": int(mask.sum()),
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
            "allow_180": int(self.allow_180),
            "obs_origin": origin,
            "centroid_dist": float(self._last_cdist),
            "loop_hit": 1.0 if loop_hit else 0.0,
            "boundary_overage": float(self._boundary_overage()),  # optional but useful
            "module_coord": tuple(int(v) for v in cur_coord.tolist()),
            "module_orig_coord": self.id2orig.get(cur_id2, tuple(cur_coord.tolist())) if cur_id2 is not None else (-1,-1,-1),
            "delta_assign": float(delta_assign),
            "delta_centroid": float(delta_centroid),
            "r_base": float(r_base),
            "r_iou": float(r_iou),
            "r_assign": float(r_assign),
            "r_centroid": float(r_centroid),
            "r_noop": float(r_noop),
            "r_loop": float(r_loop),
            "r_bound": float(r_bound),
            "reward_total": float(reward),
            "r_improve": float(r_improve),
            "r_assign_local" : float(r_assign_local),
            "improve_iou": 1.0 if improve_iou else 0.0,
            "improve_assign": 1.0 if improve_assign else 0.0,
            "improve_centroid": 1.0 if improve_centroid else 0.0,
            "local_goal": g if g else (-1,-1,-1),
            "local_dist_before": float(d_before) if d_before is not None else -1.0,
            "local_dist_after":  float(d_after)  if d_after  is not None else -1.0,
            "reward": reward,
            "nonnoop_improving_avail": 1.0 if has_improving else 0.0,
            "nonnoop_avail": nonnoop_avail,
            "neighbor_match": float(neighbor_match),
            "snap_hit": 1.0 if snap_hit else 0.0,
            **getattr(self, "_last_graph_diags", {}),
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
