# Rllib Example for single and multi-agent training for GodotRL with onnx export,
# needs rllib_config.yaml in the same folder or --config_file argument specified to work.
# Also, needs the AIController to return metrics in the get_info()

import argparse
import os
import pathlib

import ray
import yaml
from ray import train, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec

from godot_rl.core.godot_env import GodotEnv
from godot_rl.wrappers.petting_zoo_wrapper import GDRLPettingZooEnv
from godot_rl.wrappers.ray_wrapper import RayVectorGodotEnv

from ray.rllib.algorithms.callbacks import DefaultCallbacks

class MetricsCallback(DefaultCallbacks):
    def on_episode_step(
            self,
            *,
            episode,
            env_runner=None,
            metrics_logger=None,
            env=None,
            env_index: int,
            rl_module=None,
            worker=None,
            base_env=None,
            policies=None,
            **kwargs,
    ) -> None:
        # Access the latest infos from the episode
        infos = episode._last_infos  # or `episode.last_info_for()`

        # If infos exist (not None/empty)
        if infos:
            for agent_id, info in infos.items():
                # When in single-agent the info dictionary key="agent0"
                # Otherwise in multi-agent, the info will have the agent number as key, e.g. keys=[0, 1, 2, ...]
                if agent_id == "agent0" or isinstance(agent_id, int):
                    # Normalize agent0 to 0 for consistent naming
                    normalized_id = 0 if agent_id == "agent0" else agent_id
                    agent_key = f"agent_{normalized_id}"

                    # Store step-wise metrics in episode.user_data
                    episode.user_data.setdefault(f"{agent_key}", []).append(info)

    def on_episode_end(
            self,
            *,
            episode,
            env_runner = None,
            metrics_logger = None,
            env = None,
            env_index: int,
            **kwargs,
    ):
        print("Episode end")
        for key in list(episode.user_data):
            if key.startswith("agent_"):
                values = episode.user_data[key]

                # extend this list with your custom metrics sent by the AIController get_info()
                for metric in ('n_collisions', 'total_cells_found', 'bytes_sent'):
                    # Extract metric from the info dict
                    metric_values = [v.get(metric, 0) for v in values]
                    episode.custom_metrics[f"{key}_{metric}"] = sum(metric_values) / len(metric_values)
                    print(f"{key}_{metric}: {sum(metric_values) / len(metric_values)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_file", default="rlib_config.yaml", type=str, help="The yaml config file")
    parser.add_argument("--restore", default=None, type=str, help="the location of a checkpoint to restore from")
    parser.add_argument(
        "--experiment_dir",
        default="logs/rllib",
        type=str,
        help="The name of the the experiment directory, used to store logs.",
    )
    args, extras = parser.parse_known_args()

    # Get config from file
    with open(args.config_file) as f:
        exp = yaml.safe_load(f)

    is_multiagent = exp["env_is_multiagent"]

    # Register env
    env_name = "godot"
    env_wrapper = None

    def env_creator(env_config):
        index = env_config.worker_index * exp["config"]["num_envs_per_worker"] + env_config.vector_index
        port = index + GodotEnv.DEFAULT_PORT
        seed = index

        if is_multiagent:
            return ParallelPettingZooEnv(GDRLPettingZooEnv(config=env_config, port=port, seed=seed))
        else:
            return RayVectorGodotEnv(config=env_config, port=port, seed=seed)

    tune.register_env(env_name, env_creator)

    policy_names = None
    num_envs = None
    tmp_env = None

    if is_multiagent:  # Make temp env to get info needed for multi-agent training config
        print("Starting a temporary multi-agent env to get the policy names")
        tmp_env = GDRLPettingZooEnv(config=exp["config"]["env_config"], show_window=False)
        policy_names = tmp_env.agent_policy_names
        print("Policy names for each Agent (AIController) set in the Godot Environment", policy_names)
    else:  # Make temp env to get info needed for setting num_workers training config
        print("Starting a temporary env to get the number of envs and auto-set the num_envs_per_worker config value")
        tmp_env = GodotEnv(env_path=exp["config"]["env_config"]["env_path"], show_window=False)
        num_envs = tmp_env.num_envs

    tmp_env.close()

    def policy_mapping_fn(agent_id: int, episode, worker, **kwargs) -> str:
        return policy_names[agent_id]

    ray.init(_temp_dir=os.path.abspath(args.experiment_dir))

    if is_multiagent:
        exp["config"]["multiagent"] = {
            "policies": {policy_name: PolicySpec() for policy_name in policy_names},
            "policy_mapping_fn": policy_mapping_fn,
        }
    else:
        exp["config"]["num_envs_per_worker"] = num_envs

    tuner = None
    if not args.restore:
        exp["config"]["callbacks"] = MetricsCallback  # add your callback to the configs
        tuner = tune.Tuner(
            trainable=exp["algorithm"],
            param_space=exp["config"],
            run_config=train.RunConfig(
                storage_path=os.path.abspath(args.experiment_dir),
                stop=exp["stop"],
                checkpoint_config=train.CheckpointConfig(checkpoint_frequency=exp["checkpoint_frequency"]),
            ),
        )
    else:
        tuner = tune.Tuner.restore(
            trainable=exp["algorithm"],
            path=args.restore,
            resume_unfinished=True,
        )
    result = tuner.fit()

    # Onnx export after training if a checkpoint was saved
    checkpoint = result.get_best_result().checkpoint

    if checkpoint:
        result_path = result.get_best_result().path
        ppo = Algorithm.from_checkpoint(checkpoint)
        if is_multiagent:
            for policy_name in set(policy_names):
                ppo.get_policy(policy_name).export_model(f"{result_path}/onnx_export/{policy_name}_onnx", onnx=12)
                print(
                    f"Saving onnx policy to {pathlib.Path(f'{result_path}/onnx_export/{policy_name}_onnx').resolve()}"
                )
        else:
            ppo.get_policy().export_model(f"{result_path}/onnx_export/single_agent_policy_onnx", onnx=12)
            print(
                f"Saving onnx policy to {pathlib.Path(f'{result_path}/onnx_export/single_agent_policy_onnx').resolve()}"
            )

