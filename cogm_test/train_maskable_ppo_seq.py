#train_maskable_ppo_seq.py
# Train on the sequential per-module env.
# pip install gymnasium stable-baselines3 sb3-contrib numpy torch tensorboard matplotlib
import argparse, os
from sb3_contrib import MaskablePPO
from custom_ppo import AuxMaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor
from argparse import BooleanOptionalAction

from ogm_env_seq import OGMEnvSeq
from callbacks import TensorboardMetricsCallback, EpisodeVideoCallback, ExtractorMonitorCallback

def _parse_tuple3(s: str):
    parts = s.split(",")
    assert len(parts)==3, "use Comma-separated triplet, e.g. 12,12,6"
    return (int(parts[0]), int(parts[1]), int(parts[2]))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--logdir", type=str, default="logs_ogm_seq")
    parser.add_argument("--model_path", type=str, default="ppo_ogm_seq.zip")
    parser.add_argument("--video_every", type=int, default=10000)
    parser.add_argument("--allow_180", action=BooleanOptionalAction, default=True)
    parser.add_argument("--shape_weight", type=float, default=0.002)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ent_coef", type=float, default=0.05)
    parser.add_argument("--noop_tie", type=float, default=0.001)
    # NEW: observation mode & window
    parser.add_argument("--obs_mode", type=str, default="voxel", choices=["voxel","graph","both"])
    parser.add_argument("--obs_window", type=str, default="12,12,6")
    parser.add_argument("--obs_pad", type=int, default=2)
    parser.add_argument("--obs_center", type=str, default="bbox_min", choices=["bbox_min","bbox_center"])
    parser.add_argument("--max_nodes", type=int, default=64)
    parser.add_argument("--max_micro_steps", type=int, default=3000)
    # NEW:
    parser.add_argument("--randomize_order", action=BooleanOptionalAction, default=True)
    parser.add_argument("--loop_penalty", type=float, default=0.005)
    parser.add_argument("--loop_window", type=int, default=5000)
    parser.add_argument("--loop_hash_size", type=int, default=20000)
    parser.add_argument("--trans_weight", type=float, default=0.05)
    parser.add_argument("--infinite_world", action=BooleanOptionalAction, default=False,
                        help="If True, disable all OOB checks; collisions only.")
    parser.add_argument("--boundary_center", type=str, default="0,0,0",
                        help="L1-soft-boundary center (x,y,z)")
    parser.add_argument("--boundary_radius", type=float, default=0.0,
                        help="L1 radius for soft boundary (0 disables)")
    parser.add_argument("--boundary_penalty", type=float, default=0.0,
                        help="Penalty weight per unit L1 overage beyond the radius")
    parser.add_argument("--eval_freq", type=int, default=5000,
        help="Run EvalCallback every N training steps")
    parser.add_argument("--eval_eps", type=int, default=5)

    parser.add_argument("--improve_bonus", type=float, default=0.3,
                    help="Tiny reward added if IoU/assignment/centroid improves on a step")
    parser.add_argument("--graph_mode", type=str, default="combined", choices=["separate","combined"])
    parser.add_argument("--goal_edge_mode", type=str, default="radius", choices=["radius","knn"])
    parser.add_argument("--goal_edge_radius", type=int, default=2)
    parser.add_argument("--goal_edge_k", type=int, default=2)
    parser.add_argument("--connect_goal_goal", action="store_true", default=False)
    parser.add_argument("--step_cost", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.99)

    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    obs_window = _parse_tuple3(args.obs_window)
    bc = _parse_tuple3(args.boundary_center)
    ogm_kwargs = {            
            'max_micro_steps': args.max_micro_steps,
            'max_local_actions': 64,
            'success_iou': args.iou,
            'step_cost': args.step_cost,
            'invalid_cost': 0.1,
            'energy_cost': 0.0,
            'sweep_steps': 4,
            'sweep_points_per_axis': 2,
            'enforce_connectivity': True,
            'allow_180': args.allow_180,
            'shape_weight': args.shape_weight,
            'patience_no_progress': args.patience,
            'noop_tie_break_penalty': args.noop_tie,
            'obs_mode': args.obs_mode,
            'obs_window': obs_window,
            'obs_pad': args.obs_pad,
            'obs_center_mode': args.obs_center,
            'max_nodes': args.max_nodes,
            'randomize_order': args.randomize_order,
            'loop_penalty': args.loop_penalty,
            'loop_window': args.loop_window,
            'loop_hash_size': args.loop_hash_size,
            'trans_weight': args.trans_weight,
            'infinite_world': args.infinite_world,
            'boundary_center': bc,
            'boundary_l1_radius': args.boundary_radius,
            'boundary_penalty': args.boundary_penalty,
            'improve_bonus': args.improve_bonus,
            # 'eval_eps': args.eval_eps,
            }

    def make_train_env():
        dims = (32,32,8)   # world box; keep large enough for your tasks
        init_modules = [(0,0,0),(1,0,0),(1,1,0),(2,1,0)]
        goal_modules = [(1,2,0),(2,2,0),(1,3,0),(2,3,0)]

        e = OGMEnvSeq(
            dims=dims,
            init_modules=init_modules,
            goal_modules=goal_modules,
            **ogm_kwargs
        )
        e = Monitor(e)
        return ActionMasker(e, lambda env: env.unwrapped.get_action_mask())

    train_env = DummyVecEnv([make_train_env])

    def make_eval_env():
        dims = (32,32,8)   # world box; keep large enough for your tasks
        init_modules = [(0,0,0),(1,0,0),(1,1,0),(2,1,0)]
        goal_modules = [(1,2,0),(2,2,0),(1,3,0),(2,3,0)]
        ekw = dict(ogm_kwargs)
        ekw["max_micro_steps"] = min(800, ogm_kwargs["max_micro_steps"])
        ekw["patience_no_progress"] = 5  # end stale episodes faster
        env = OGMEnvSeq(dims=dims, init_modules=init_modules, goal_modules=goal_modules, **ekw)
        env = Monitor(env)
        return ActionMasker(env, lambda e: e.unwrapped.get_action_mask())

    eval_env = DummyVecEnv([make_eval_env])    

    # eval_env = DummyVecEnv([make_train_env])

    policy_kwargs = {}
    if args.obs_mode in ("graph", "both"):
        from gnn_extractor import GraphExtractor
        policy_kwargs = dict(
            features_extractor_class=GraphExtractor,
            features_extractor_kwargs=dict(
                node_dim=3, hidden_dim=128, gcn_layers =2, out_dim=256
            ),
        )

    # model = MaskablePPO(
    model = AuxMaskablePPO(
        policy="MultiInputPolicy",
        env=train_env,
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        n_steps=2048,
        batch_size=2048,
        learning_rate=3e-4,
        gamma=0.98,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=args.logdir,
        aux_weight=0.1,
    )

    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=args.logdir,
        log_path=args.logdir,
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
    )

    def make_raw_env():
        # Unmasked env for video
        dims = (32,32,8)
        init_modules = [(0,0,0),(1,0,0),(1,1,0),(2,1,0)]
        goal_modules = [(1,2,0),(2,2,0),(1,3,0),(2,3,0)]
        return OGMEnvSeq(
            dims=dims,
            init_modules=init_modules,
            goal_modules=goal_modules,
            **ogm_kwargs
        )

    tb_cb = TensorboardMetricsCallback()
    vid_cb = EpisodeVideoCallback(make_env_fn=make_raw_env, every_timesteps=args.video_every, max_steps=200, fps=4, video_dir=os.path.join(args.logdir, "videos"), write_gif=False)
    gnn_cb = ExtractorMonitorCallback(every_updates=1)  # log after each PPO update block

    callback = CallbackList([eval_cb, tb_cb, vid_cb, gnn_cb])
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.model_path)
    print("Saved model to", args.model_path)

if __name__ == "__main__":
    main()
