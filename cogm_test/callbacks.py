#callbacks.py
from __future__ import annotations
from typing import Callable, Optional
import numpy as np
import os
from collections import defaultdict
import imageio
import torch as th
import math
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.monitor import Monitor

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

def _auto_bounds(modules, goals, margin=1.0):
    pts = []
    for c in modules: pts += [c, (c[0]+1,c[1]+1,c[2]+1)]
    for c in goals: pts += [c, (c[0]+1,c[1]+1,c[2]+1)]
    if not pts: return (-0.25,0.25),(-0.25,0.25),(-0.25,0.25)
    P = np.array(pts, float)
    mins = P.min(axis=0) - margin
    maxs = P.max(axis=0) + margin
    return (mins[0],maxs[0]), (mins[1],maxs[1]), (mins[2],maxs[2])

def _pad16(img):
    import numpy as np
    h, w, c = img.shape
    H = ((h + 15) // 16) * 16
    W = ((w + 15) // 16) * 16
    if H == h and W == w:
        return img
    out = np.zeros((H, W, c), dtype=img.dtype)
    out[:h, :w] = img
    return out

def _draw_frame(modules, goal_modules, act=None, obs_origin=None, overlay_lines=None, figsize=(6.4,6.4), dpi=100):
    fig = plt.figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(projection='3d')

    (xl,xh),(yl,yh),(zl,zh) = _auto_bounds(modules, goal_modules, margin=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_xlim(xl, xh); ax.set_ylim(yl, yh); ax.set_zlim(zl, zh)
    ax.set_box_aspect((xh-xl, yh-yl, zh-zl))
    ax.grid(True)

    for c in goal_modules:
        faces = _cube_faces_from_min_corner(c, 1.0)
        pc = Poly3DCollection(faces, alpha=0.08, linewidths=1); ax.add_collection3d(pc)
        for f in faces:
            xs, ys, zs = zip(*(f + [f[0]])); ax.plot(xs, ys, zs, linewidth=1)

    for c in modules:
        faces = _cube_faces_from_min_corner(c, 1.0)
        pc = Poly3DCollection(faces, alpha=0.25, linewidths=1); ax.add_collection3d(pc)
        for f in faces:
            xs, ys, zs = zip(*(f + [f[0]])); ax.plot(xs, ys, zs, linewidth=1)

    if act is not None:
        p = np.array(act.edge_center, dtype=float)
        v = _AXIS_VEC[str(act.edge_axis)]
        length = max(xh-xl, yh-yl, zh-zl) * 0.6
        p1 = p - v*length*0.5; p2 = p + v*length*0.5
        ax.plot([p1[0], p2[0]],[p1[1], p2[1]],[p1[2], p2[2]], linewidth=3)
        ax.scatter([p[0]],[p[1]],[p[2]], s=40)

    # overlays (origin + reward breakdown lines)
    y0 = 0.95
    if obs_origin is not None:
        ax.text2D(0.02, y0, f"obs_origin={obs_origin}", transform=ax.transAxes, fontsize=9)
        y0 -= 0.05
    if overlay_lines:
        for line in overlay_lines:
            ax.text2D(0.02, y0, line, transform=ax.transAxes, fontsize=9)
            y0 -= 0.05

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = buf[..., :3].copy()
    plt.close(fig)
    return img

class TensorboardMetricsCallback(BaseCallback):
    def __init__(self, log_per_module: bool = True):
        super().__init__()
        self.log_per_module = log_per_module
        self._writer: Optional[SummaryWriter] = None
        self._cum_rej = defaultdict(lambda: defaultdict(int))

    def _on_training_start(self) -> None:
        log_dir = self.logger.get_dir()
        if log_dir is not None:
            self._writer = SummaryWriter(log_dir)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos: return True
        info = infos[0]

        # ... keep your existing logger.record(...) code ...

        if "local_valid_actions" in info:
            self.logger.record("train/valid_actions", float(info["local_valid_actions"]))
        elif "valid_actions" in info:
            self.logger.record("train/valid_actions", float(info["valid_actions"]))

        for key in ("reward", "iou","nonnoop_count","allowed_nonnoop_mean","cur_index","num_modules",
                    "at_original","neighbor_fine","nonnoop_taken","allow_180", "centroid_dist", 
                    "loop_hit", "r_improve","improve_iou","improve_assign","improve_centroid",
                    "delta_centroid","delta_assign", "nonnoop_taken", "r_base",  "r_iou", "r_assign",
                    "r_centroid", "r_noop", "r_loop", "r_bound"):
            if key in info:
                val = info[key]
                if isinstance(val, bool): val = 1.0 if val else 0.0
                self.logger.record(f"train/{key}", float(val))
        rc = info.get("rej_counts")
        if isinstance(rc, dict):
            for k,v in rc.items():
                self.logger.record(f"train/reject_{k}", float(v))

        if not (self.log_per_module and self._writer is not None):
            return True

        rc = info.get("rej_counts")
        mid = info.get("module_id", -1)
        mcoord = info.get("module_coord", None)
        morig = info.get("module_orig_coord", None)

        # STABLE label: by id, optionally with original coord (constant)
        if mid is not None and mid >= 0:
            if morig is not None:
                mlabel = f"id{mid}_orig{morig}"
            else:
                mlabel = f"id{mid}"
        else:
            mlabel = "id-1"

        # Log current coordinate as separate scalars (won't create new series)
        if isinstance(mcoord, (tuple, list)) and len(mcoord) == 3:
            self._writer.add_scalar(f"module_coord_x/{mlabel}", mcoord[0], self.num_timesteps)
            self._writer.add_scalar(f"module_coord_y/{mlabel}", mcoord[1], self.num_timesteps)
            self._writer.add_scalar(f"module_coord_z/{mlabel}", mcoord[2], self.num_timesteps)

        # Per-module rejection reasons (stable series)
        if isinstance(rc, dict):
            for reason, cnt in rc.items():
                self._writer.add_scalar(f"rejections_per_module/{mlabel}/{reason}", int(cnt), self.num_timesteps)
                self._cum_rej[mlabel][reason] += int(cnt)
            # cumulative view
            for reason, total in self._cum_rej[mlabel].items():
                self._writer.add_scalar(f"rejections_cum_per_module/{mlabel}/{reason}", int(total), self.num_timesteps)

        self._writer.flush()
        return True

# class TensorboardMetricsCallback(BaseCallback):
#     def _on_step(self) -> bool:


class EpisodeVideoCallback(BaseCallback):
    def __init__(
        self,
        make_env_fn: Callable[[], "gym.Env"],
        every_timesteps: int = 10000,
        max_steps: int = 80,
        fps: int = 4,
        video_dir: Optional[str] = "videos",
        log_to_tensorboard: bool = True,
        write_gif: bool = True,
        write_mp4: bool = True,
    ):
        super().__init__()
        self.make_env_fn = make_env_fn
        self.every_timesteps = int(every_timesteps)
        self.max_steps = int(max_steps)
        self.fps = int(fps)
        self.video_dir = video_dir
        self.log_to_tensorboard = log_to_tensorboard
        self.write_gif = bool(write_gif)
        self.write_mp4 = bool(write_mp4)
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
            env = Monitor(ActionMasker(env, lambda e: e.unwrapped.get_action_mask()))

        obs, info = env.reset()
        frames = []
        modules = set(map(tuple, raw_env.grid.modules))
        frames.append(_draw_frame(modules, raw_env.goal_modules, act=None, 
                                  obs_origin=info.get("obs_origin"),
                                 overlay_lines=["t=-1 (reset)"]))

        for t in range(self.max_steps):
            mask = raw_env.get_action_mask()
            action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
            act = self._get_impending_act(raw_env, action)
            modules = set(map(tuple, raw_env.grid.modules))
            obs, reward, terminated, truncated, info = env.step(int(action))
            ol = [
                f"t={t} a={int(action)} IoU={info.get('iou',0):.3f}",
                "r_step={:+.3f} r_iou={:+.3f} r_assign={:+.3f}".format(
                    float(info.get('r_step',0.0)),
                    float(info.get('r_iou',0.0)),
                    float(info.get('r_assign',0.0))),
                "r_trans={:+.3f} r_drift={:+.3f} r_improve={:+.3f} r_noop={:+.3f}".format(
                    float(info.get('r_trans',0.0)),
                    float(info.get('r_drift',0.0)),
                    float(info.get('r_improve',0.0)),
                    float(info.get('r_noop',0.0))),
                "r_assign_local={:+.3f}".format(float(info.get('r_assign_local',0.0))),
                "Δassign={:+.2f} Δcentroid={:+.2f} R={:+.3f}".format(
                    float(info.get('delta_assign',0.0)),
                    float(info.get('delta_centroid',0.0)),
                    float(reward)),
            ]
            modules = set(map(tuple, raw_env.grid.modules))
            frames.append(_draw_frame(modules, raw_env.goal_modules, act=act, obs_origin=info.get("obs_origin"),
                                     overlay_lines=ol))
            if terminated or truncated:
                break

        frames = [_pad16(f) for f in frames]
        vid = np.stack(frames, axis=0)  # T,H,W,C
        T, H, W, C = vid.shape
        vid_t = np.transpose(vid, (0,3,1,2))  # T,C,H,W
        vid_t = np.expand_dims(vid_t, 0)      # 1,T,C,H,W

        if self._writer is not None and self.log_to_tensorboard:
            try:
                self._writer.add_video(f"episode_video/step_{self.num_timesteps}", vid_t.astype(np.float32)/255.0, global_step=self.num_timesteps, fps=self.fps)
                self._writer.flush()
            except Exception:
                try:
                    self._writer.add_image(f"episode_video/step_{self.num_timesteps}/first_frame", vid_t[0,0], global_step=self.num_timesteps)
                    self._writer.flush()
                except Exception:
                    pass

        if self.video_dir is not None:
            os.makedirs(self.video_dir, exist_ok=True)
            if self.write_gif:
                try:
                    out_path = os.path.join(self.video_dir, f"episode_{self.num_timesteps}.gif")
                    imageio.mimsave(out_path, [f for f in frames], fps=self.fps)
                except Exception:
                    pass
            if self.write_mp4:
                mp4_path = os.path.join(self.video_dir, f"episode_{self.num_timesteps}.mp4")
                wrote = False
                try:
                    import imageio
                    writer = imageio.get_writer(mp4_path, fps=self.fps, codec='libx264', quality=8, macro_block_size=1)  # needs imageio-ffmpeg
                    for f in frames: writer.append_data(f)
                    writer.close()
                    wrote = True
                except Exception:
                    pass

    def _on_step(self) -> bool:
        if self._should_record():
            self._record_once()
        return True

    def _on_training_end(self) -> None:
        if self._writer is not None:
            self._writer.flush(); self._writer.close()


from typing import Dict
import torch
import torch.nn.functional as F
try:
    from custom_ppo import _centroid_l1_target
except Exception:
    _centroid_l1_target = None

class ExtractorMonitorCallback(BaseCallback):
    """
    Logs GNN feature norms and (optionally) the auxiliary MSE on a minibatch
    from the latest rollout, once per PPO update block.
    """
    def __init__(self, every_updates: int = 1):
        super().__init__()
        self.every_updates = int(every_updates)

    # REQUIRED by BaseCallback (abstract)
    def _on_step(self) -> bool:
        # no per-env-step work
        return True

    def _on_update(self) -> bool:
        # Called after each PPO update block
        if (getattr(self.model, "_n_updates", 0) % self.every_updates) != 0:
            return True

        policy = self.model.policy
        fe = getattr(policy, "features_extractor", None)
        if fe is None:
            return True

        # Take a small slice from the rollout buffer to evaluate features
        rb = self.model.rollout_buffer
        n = min(256, rb.buffer_size)
        if n <= 0:
            return True

        # Build a dict of torch tensors on the right device
        obs_batch: Dict[str, torch.Tensor] = {}
        for k, arr in rb.observations.items():
            # take first n, convert to torch on policy device
            obs_batch[k] = torch.as_tensor(arr[:n], device=policy.device)

        with torch.no_grad():
            feats = policy.extract_features(obs_batch)  # [n, F]
            feat_norm = feats.norm(dim=-1).mean().item()
            feat_std = feats.std(dim=0).mean().item()

            self.logger.record("gnn/feat_norm_mean", float(feat_norm))
            self.logger.record("gnn/feat_std_mean", float(feat_std))

            # If the extractor has an aux head, compute a quick MSE snapshot
            if hasattr(fe, "aux_head"):
                aux_pred = fe.aux_head(feats).squeeze(-1)  # [n]
                if _centroid_l1_target is not None:
                    tgt = _centroid_l1_target(obs_batch).to(aux_pred.device)  # [n]
                    aux_mse = F.mse_loss(aux_pred, tgt).item()
                    self.logger.record("gnn/aux_mse_eval", float(aux_mse))

        return True