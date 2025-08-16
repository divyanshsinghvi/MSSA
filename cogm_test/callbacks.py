from __future__ import annotations
from typing import Callable, Optional
import numpy as np
import os

from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from sb3_contrib.common.wrappers import ActionMasker
except Exception:
    ActionMasker = None  # type: ignore

_AXIS_VEC = {'x': np.array([1.0,0.0,0.0]), 'y': np.array([0.0,1.0,0.0]), 'z': np.array([0.0,0.0,1.0])}

def _cube_faces_from_min_corner(origin, size=1.0):
    x, y, z = origin; s = size
    r = [x, x + s]; t = [y, y + s]; u = [z, z + s]
    return [
        [(r[0], t[0], u[0]), (r[1], t[0], u[0]), (r[1], t[1], u[0]), (r[0], t[1], u[0])],
        [(r[0], t[0], u[1]), (r[1], t[0], u[1]), (r[1], t[1], u[1]), (r[0], t[1], u[1])],
        [(r[0], t[0], u[0]), (r[1], t[0], u[0]), (r[1], t[0], u[1]), (r[0], t[0], u[1])],
        [(r[0], t[1], u[0]), (r[1], t[1], u[0]), (r[1], t[1], u[1]), (r[0], t[1], u[1])],
        [(r[0], t[0], u[0]), (r[0], t[1], u[0]), (r[0], t[1], u[1]), (r[0], t[0], u[1])],
        [(r[1], t[0], u[0]), (r[1], t[1], u[0]), (r[1], t[1], u[1]), (r[1], t[0], u[1])],
    ]

def _draw_frame(dims, modules, goal_modules, act=None, figsize=(6,6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = dims
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_xlim(-0.25, X+0.25); ax.set_ylim(-0.25, Y+0.25); ax.set_zlim(-0.25, Z+0.25)
    ax.set_box_aspect((X, Y, Z))
    ax.grid(True)

    for c in goal_modules:
        faces = _cube_faces_from_min_corner(c, 1.0)
        pc = Poly3DCollection(faces, alpha=0.08, linewidths=1)
        ax.add_collection3d(pc)
        for f in faces:
            xs, ys, zs = zip(*(f + [f[0]]))
            ax.plot(xs, ys, zs, linewidth=1)

    for c in modules:
        faces = _cube_faces_from_min_corner(c, 1.0)
        pc = Poly3DCollection(faces, alpha=0.25, linewidths=1)
        ax.add_collection3d(pc)
        for f in faces:
            xs, ys, zs = zip(*(f + [f[0]]))
            ax.plot(xs, ys, zs, linewidth=1)

    if act is not None:
        p = np.array(act.edge_center, dtype=float)
        v = _AXIS_VEC[str(act.edge_axis)]
        length = max(dims) * 0.6
        p1 = p - v*length*0.5
        p2 = p + v*length*0.5
        ax.plot([p1[0], p2[0]],[p1[1], p2[1]],[p1[2], p2[2]], linewidth=3)
        ax.scatter([p[0]],[p[1]],[p[2]], s=40)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = buf[..., :3].copy()
    plt.close(fig)
    return img

class TensorboardMetricsCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos: return True
        info = infos[0]
        for key in ("iou","valid_actions","nonnoop_count","allowed_nonnoop_mean","cur_index","num_modules","at_original","neighbor_fine","nonnoop_taken"):
            if key in info:
                val = info[key]
                if isinstance(val, bool): val = 1.0 if val else 0.0
                self.logger.record(f"train/{key}", float(val))
        # rejection reason counts
        rc = info.get("rej_counts")
        if isinstance(rc, dict):
            for k,v in rc.items():
                self.logger.record(f"train/reject_{k}", float(v))
        return True

class EpisodeVideoCallback(BaseCallback):
    def __init__(
        self,
        make_env_fn: Callable[[], "gym.Env"],
        every_timesteps: int = 10000,
        max_steps: int = 80,
        fps: int = 4,
        video_dir: Optional[str] = "videos",
        log_to_tensorboard: bool = True,
    ):
        super().__init__()
        self.make_env_fn = make_env_fn
        self.every_timesteps = int(every_timesteps)
        self.max_steps = int(max_steps)
        self.fps = int(fps)
        self.video_dir = video_dir
        self.log_to_tensorboard = log_to_tensorboard
        self._last_ts = 0
        self._writer: Optional[SummaryWriter] = None

    def _on_training_start(self) -> None:
        log_dir = self.logger.get_dir()
        if self.log_to_tensorboard and log_dir is not None:
            self._writer = SummaryWriter(log_dir)
        if self.video_dir is not None:
            os.makedirs(self.video_dir, exist_ok=True)

    def _should_record(self) -> bool:
        return (self.num_timesteps - self._last_ts) >= self.every_timesteps

    def _get_impending_act(self, raw_env, action_int):
        try: idx = int(action_int)
        except Exception: return None
        table = getattr(raw_env, "_local_actions", None)
        if table is not None and isinstance(table, (list, tuple)):
            if 0 <= idx < len(table): return table[idx]
            return None
        table = getattr(raw_env, "_action_table", None)
        if table is not None and isinstance(table, (list, tuple)):
            if 0 <= idx < len(table): return table[idx]
        return None

    def _record_once(self):
        self._last_ts = self.num_timesteps
        env = self.make_env_fn()
        raw_env = env
        if ActionMasker is not None:
            env = ActionMasker(env, lambda env: env.get_action_mask())

        obs, info = env.reset()
        frames = []
        modules = set(map(tuple, raw_env.grid.modules))
        frames.append(_draw_frame(raw_env.dims, modules, raw_env.goal_modules, act=None))

        for t in range(self.max_steps):
            mask = raw_env.get_action_mask()
            action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
            act = self._get_impending_act(raw_env, action)
            modules = set(map(tuple, raw_env.grid.modules))
            frames.append(_draw_frame(raw_env.dims, modules, raw_env.goal_modules, act=act))
            obs, reward, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                modules = set(map(tuple, raw_env.grid.modules))
                frames.append(_draw_frame(raw_env.dims, modules, raw_env.goal_modules, act=None))
                break

        vid = np.stack(frames, axis=0)  # T,H,W,C
        T, H, W, C = vid.shape
        vid_t = np.transpose(vid, (0,3,1,2))  # T,C,H,W
        vid_t = np.expand_dims(vid_t, 0)      # 1,T,C,H,W

        if self._writer is not None and self.log_to_tensorboard:
            try:
                self._writer.add_video(f"episode_video/step_{self.num_timesteps}", vid_t.astype(np.float32)/255.0, global_step=self.num_timesteps, fps=self.fps)
                self._writer.flush()
            except Exception:
                # Fallback: log first frame as an image
                try:
                    self._writer.add_image(f"episode_video/step_{self.num_timesteps}/first_frame", vid_t[0,0], global_step=self.num_timesteps)
                    self._writer.flush()
                except Exception:
                    pass

        if self.video_dir is not None:
            try:
                import imageio
                out_path = os.path.join(self.video_dir, f"episode_{self.num_timesteps}.gif")
                imageio.mimsave(out_path, [f for f in frames], fps=self.fps)
            except Exception:
                pass

    def _on_step(self) -> bool:
        if self._should_record():
            self._record_once()
        return True

    def _on_training_end(self) -> None:
        if self._writer is not None:
            self._writer.flush(); self._writer.close()
