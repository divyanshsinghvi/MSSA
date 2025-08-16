
# train_maskable_ppo_seq.py
# Train on the sequential per-module env.
# pip install gymnasium stable-baselines3 sb3-contrib numpy torch tensorboard matplotlib

import argparse, os
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from ogm_env_seq import OGMEnvSeq
from callbacks import TensorboardMetricsCallback, EpisodeVideoCallback

def make_env():
    dims = (4,4,2)
    init_modules = [(0,0,0),(1,0,0),(1,1,0),(2,1,0)]
    goal_modules = [(1,2,0),(2,2,0),(1,3,0),(2,3,0)]
    env = OGMEnvSeq(
        dims=dims,
        init_modules=init_modules,
        goal_modules=goal_modules,
        max_micro_steps=3000,
        max_local_actions=64,
        success_iou=0.99,
        step_cost=0.01,
        invalid_cost=0.1,
        energy_cost=0.0,
        sweep_steps=12,
        sweep_points_per_axis=4,
        enforce_connectivity=True,
    )
    env = ActionMasker(env, lambda env: env.get_action_mask())
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--logdir", type=str, default="logs_ogm_seq")
    parser.add_argument("--model_path", type=str, default="ppo_ogm_seq.zip")
    parser.add_argument("--video_every", type=int, default=10000, help="record an eval episode every N steps")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=2048,
        batch_size=2048,
        learning_rate=3e-4,
        gamma=0.98,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=args.logdir,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.logdir,
        log_path=args.logdir,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
    )

    tb_cb = TensorboardMetricsCallback()
    # Video callback expects the raw env factory; reuse make_env but unwrap ActionMasker inside callback
    from ogm_env_seq import OGMEnvSeq as RawEnvType
    def make_raw_env():
        dims = (4,4,2)
        init_modules = [(0,0,0),(1,0,0),(1,1,0),(2,1,0)]
        goal_modules = [(1,2,0),(2,2,0),(1,3,0),(2,3,0)]
        return RawEnvType(
            dims=dims,
            init_modules=init_modules,
            goal_modules=goal_modules,
            max_micro_steps=3000,
            max_local_actions=64,
            success_iou=0.99,
            step_cost=0.01,
            invalid_cost=0.1,
            energy_cost=0.0,
            sweep_steps=12,
            sweep_points_per_axis=4,
            enforce_connectivity=True,
        )
    vid_cb = EpisodeVideoCallback(make_env_fn=make_raw_env, every_timesteps=args.video_every, max_steps=200, fps=4, video_dir=os.path.join(args.logdir, "videos"))

    callback = CallbackList([eval_cb, tb_cb, vid_cb])

    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.model_path)
    print("Saved model to", args.model_path)

if __name__ == "__main__":
    main()
