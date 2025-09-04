#ogm.py
from __future__ import annotations
from typing import List, Tuple, Set, Dict, Any, Iterable
import numpy as np

Vec3 = Tuple[int, int, int]

def face_adjacent(a: Vec3, b: Vec3) -> bool:
    return sum(abs(ax - bx) for ax, bx in zip(a, b)) == 1

class EdgeAction:
    def __init__(self, m: Vec3, n: Vec3, edge_axis: str, edge_center: Tuple[float,float,float],
                 theta_deg: int, end_min_corner: Vec3):
        self.m = tuple(m)
        self.n = tuple(n)
        self.edge_axis = edge_axis  # 'x' | 'y' | 'z'
        self.edge_center = tuple(float(c) for c in edge_center)
        self.theta_deg = int(theta_deg)  # ±90 or ±180 (depending on config)
        self.end_min_corner = tuple(end_min_corner)

    def __repr__(self) -> str:
        return (f"EdgeAction(m={self.m}, n={self.n}, axis={self.edge_axis}, center={self.edge_center}, "
                f"theta={self.theta_deg}, end={self.end_min_corner})")

class OccupancyGridMap:
    def __init__(self, dims: Tuple[int,int,int], init_modules: Iterable[Vec3], obstacles: Iterable[Vec3] = (),
                  infinite_world: bool = False):
        self.dims = tuple(int(x) for x in dims)
        self.modules: Set[Vec3] = set(map(tuple, init_modules))
        self.obstacles: Set[Vec3] = set(map(tuple, obstacles))
        self.infinite_world = bool(infinite_world)
    
    # ---------- windowed observation (NEW) ----------
    @staticmethod
    def _fill_window(coords: Iterable[Vec3], origin: Vec3, size: Tuple[int,int,int]) -> np.ndarray:
        ox, oy, oz = origin
        X, Y, Z = size
        g = np.zeros((X, Y, Z), dtype=np.uint8)
        for (x,y,z) in coords:
            ix, iy, iz = x - ox, y - oy, z - oz
            if 0 <= ix < X and 0 <= iy < Y and 0 <= iz < Z:
                g[ix, iy, iz] = 1
        return g

    def occupancy_window(self, origin: Vec3, size: Tuple[int,int,int]) -> np.ndarray:
        return self._fill_window(self.modules, origin, size)

    def goal_window(self, goal_modules: Iterable[Vec3], origin: Vec3, size: Tuple[int,int,int]) -> np.ndarray:
        return self._fill_window(goal_modules, origin, size)

    # ---------- legacy full-grid builders ----------
    def occupancy_grid(self) -> np.ndarray:
        X,Y,Z = self.dims
        g = np.zeros((X,Y,Z), dtype=np.uint8)
        for (x,y,z) in self.modules:
            if 0 <= x < X and 0 <= y < Y and 0 <= z < Z:
                g[x,y,z] = 1
        return g

    def goal_grid(self, goal_modules: Iterable[Vec3]) -> np.ndarray:
        X,Y,Z = self.dims
        g = np.zeros((X,Y,Z), dtype=np.uint8)
        for (x,y,z) in goal_modules:
            if 0 <= x < X and 0 <= y < Y and 0 <= z < Z:
                g[x,y,z] = 1
        return g

    # ---------- topology ----------
    def neighbors_face(self, cell: Vec3) -> Set[Vec3]:
        (x,y,z) = cell
        cand = [(x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)]
        return {c for c in cand if c in self.modules}

    def _connected(self, cells: Set[Vec3]) -> bool:
        if not cells: return True
        from collections import deque
        start = next(iter(cells))
        seen = {start}
        q = deque([start])
        while q:
            (x,y,z) = q.popleft()
            for nx,ny,nz in [(x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)]:
                c = (nx,ny,nz)
                if c in cells and c not in seen:
                    seen.add(c); q.append(c)
        return len(seen) == len(cells)

    # ---------- hinge geometry ----------
    def _edge_centers_and_axes(self, m: Vec3, n: Vec3) -> List[Tuple[str, Tuple[float,float,float]]]:
        (x,y,z) = m
        (nx,ny,nz) = n
        dx,dy,dz = nx - x, ny - y, nz - z
        centers: List[Tuple[str, Tuple[float,float,float]]] = []
        if abs(dx) == 1 and dy == 0 and dz == 0:
            fx = x + (1 if dx == 1 else 0)
            centers += [('y',(fx, y+0.5, z+0.0)), ('y',(fx, y+0.5, z+1.0)),
                        ('z',(fx, y+0.0, z+0.5)), ('z',(fx, y+1.0, z+0.5))]
        elif abs(dy) == 1 and dx == 0 and dz == 0:
            fy = y + (1 if dy == 1 else 0)
            centers += [('x',(x+0.5, fy, z+0.0)), ('x',(x+0.5, fy, z+1.0)),
                        ('z',(x+0.0, fy, z+0.5)), ('z',(x+1.0, fy, z+0.5))]
        elif abs(dz) == 1 and dx == 0 and dy == 0:
            fz = z + (1 if dz == 1 else 0)
            centers += [('x',(x+0.5, y+0.0, fz)), ('x',(x+0.5, y+1.0, fz)),
                        ('y',(x+0.0, y+0.5, fz)), ('y',(x+1.0, y+0.5, fz))]
        return centers

    def _rotate_point_about_axis(self, p: Tuple[float,float,float],
                                 center: Tuple[float,float,float],
                                 axis: str, ang: float) -> Tuple[float,float,float]:
        px,py,pz = p
        cx,cy,cz = center
        vx,vy,vz = px-cx, py-cy, pz-cz
        cA, sA = np.cos(ang), np.sin(ang)
        if axis == 'x':
            vy2 = cA*vy - sA*vz; vz2 = sA*vy + cA*vz; vx2 = vx
        elif axis == 'y':
            vx2 = cA*vx + sA*vz; vz2 = -sA*vx + cA*vz; vy2 = vy
        elif axis == 'z':
            vx2 = cA*vx - sA*vy; vy2 = sA*vx + cA*vy; vz2 = vz
        else:
            raise ValueError("axis must be 'x','y','z'")
        return (cx+vx2, cy+vy2, cz+vz2)

    def _rotated_min_corner(self, m: Vec3, axis: str, center: Tuple[float,float,float], theta_deg: int) -> Vec3:
        x,y,z = m
        verts = [(x+dx, y+dy, z+dz) for dx in (0,1) for dy in (0,1) for dz in (0,1)]
        ang = np.deg2rad(theta_deg)
        rverts = [self._rotate_point_about_axis(v, center, axis, ang) for v in verts]
        rx = min(v[0] for v in rverts); ry = min(v[1] for v in rverts); rz = min(v[2] for v in rverts)
        eps = 1e-6
        return (int(np.floor(rx+eps)), int(np.floor(ry+eps)), int(np.floor(rz+eps)))

    # ---------- swept volume ----------
    def swept_volume_reason(self, m: Vec3, axis: str, center: Tuple[float,float,float], theta_deg: int,
                            static_minus_m: Set[Vec3], sweep_steps: int, sweep_points_per_axis: int) -> Tuple[bool,str]:
        x,y,z = m
        eps = 1e-3
        grid = np.linspace(eps, 1.0-eps, sweep_points_per_axis)
        pts = [(x+ix, y+iy, z+iz) for ix in grid for iy in grid for iz in grid]
        angles = np.linspace(0.0, np.deg2rad(theta_deg), sweep_steps+1)[1:-1] if sweep_steps >= 2 else []
        X,Y,Z = self.dims
        for ang in angles:
            for p in pts:
                q = self._rotate_point_about_axis(p, center, axis, ang)
                ix,iy,iz = int(np.floor(q[0])), int(np.floor(q[1])), int(np.floor(q[2]))
                if not self.infinite_world:
                    if ix < 0 or iy < 0 or iz < 0 or ix >= X or iy >= Y or iz >= Z:
                        return (False, "mid_rotation_oob")
                if (ix,iy,iz) in static_minus_m or (ix,iy,iz) in self.obstacles:
                    return (False, "swept_collision")
        return (True, "ok")

    # ---------- enumeration ----------
    def enumerate_edge_actions_for_with_reasons(
        self, m: Vec3, sweep_steps: int = 8, sweep_points_per_axis: int = 3,
        enforce_connectivity: bool = True, allow_180: bool = True
    ):
        actions: List[EdgeAction] = []
        rejections: List[Dict[str, Any]] = []
        if m not in self.modules:
            return actions, rejections
        static = set(self.modules)
        neighs = self.neighbors_face(m)
        if not neighs:
            return actions, rejections

        if enforce_connectivity and not self._connected(static - {m}):
            rejections.append(dict(mover=m, reason="disconnects_static_without_m"))
            return actions, rejections

        for n in neighs:
            for (axis, center) in self._edge_centers_and_axes(m, n):
                thetas = (+90,-90) + ((+180,-180) if allow_180 else ())
                for theta in thetas:
                    end_c = self._rotated_min_corner(m, axis, center, theta)
                    if end_c == m:
                        rejections.append(dict(edge_axis=axis, edge_center=center, theta_deg=theta, reason="end_same_as_start"))
                        continue
                    X,Y,Z = self.dims
                    if not self.infinite_world:
                        if not (0 <= end_c[0] < X and 0 <= end_c[1] < Y and 0 <= end_c[2] < Z):
                            rejections.append(dict(edge_axis=axis, edge_center=center, theta_deg=theta, reason="end_oob")); continue
                    if end_c in static or end_c in self.obstacles:
                        rejections.append(dict(edge_axis=axis, edge_center=center, theta_deg=theta, reason="end_cell_occupied")); continue
                    static_minus = static - {m}
                    if not any(face_adjacent(end_c, s) for s in static_minus):
                        rejections.append(dict(edge_axis=axis, edge_center=center, theta_deg=theta, reason="end_not_face_adjacent")); continue
                    ok, why = self.swept_volume_reason(m, axis, center, theta, static_minus, sweep_steps, sweep_points_per_axis)
                    if not ok:
                        rejections.append(dict(edge_axis=axis, edge_center=center, theta_deg=theta, reason=why)); continue
                    if enforce_connectivity and not self._connected(static_minus | {end_c}):
                        rejections.append(dict(edge_axis=axis, edge_center=center, theta_deg=theta, reason="disconnects_after_move")); continue
                    actions.append(EdgeAction(m=m, n=n, edge_axis=axis, edge_center=center, theta_deg=theta, end_min_corner=end_c))
        return actions, rejections

    def enumerate_edge_actions_for(self, m: Vec3, sweep_steps: int = 8, sweep_points_per_axis: int = 3,
                                   enforce_connectivity: bool = True, allow_180: bool = True) -> List[EdgeAction]:
        acts, _ = self.enumerate_edge_actions_for_with_reasons(m, sweep_steps, sweep_points_per_axis, enforce_connectivity, allow_180)
        return acts

    def enumerate_edge_actions(self, sweep_steps: int = 8, sweep_points_per_axis: int = 3,
                               enforce_connectivity: bool = True, allow_180: bool = True) -> List[EdgeAction]:
        out: List[EdgeAction] = []
        for m in sorted(self.modules):
            out.extend(self.enumerate_edge_actions_for(m, sweep_steps, sweep_points_per_axis, enforce_connectivity, allow_180))
        return out

    # ---------- apply ----------
    def apply_action(self, act: EdgeAction) -> None:
        if act.m not in self.modules:
            return
        self.modules.remove(act.m)
        self.modules.add(act.end_min_corner)

    # ---------- metrics ----------
    @staticmethod
    def iou(a: Iterable[Vec3], b: Iterable[Vec3]) -> float:
        A = set(map(tuple, a)); B = set(map(tuple, b))
        if not A and not B: return 1.0
        inter = len(A & B); uni = len(A | B)
        return inter / max(1, uni)
