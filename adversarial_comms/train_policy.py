import argparse
import collections.abc
import yaml
import json
import os
import ray

import numpy as np

from pathlib import Path
from ray import tune
from ray.rllib.utils import try_import_torch
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print, DEFAULT_LOGGERS, TBXLogger
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.rllib.agents.callbacks import DefaultCallbacks

from .environments.coverage import CoverageEnv
from .environments.path_planning import PathPlanningEnv
from .models.adversarial import AdversarialModel
from .trainers.multiagent_ppo import MultiPPOTrainer

torch, _ = try_import_torch()

def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def trial_dirname_creator(trial):
    return str(trial) #f"{ray.tune.trial.date_str()}_{trial}"

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def check_file(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)

def get_config_base():
    return Path(os.path.dirname(os.path.realpath(__file__))) / "config"

class EvaluationCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        episode.user_data["reward_greedy"] = []
        episode.user_data["reward_coop"] = []

    def on_episode_step(self, worker, base_env, episode, **kwargs):
        ep_info = episode.last_info_for()
        if ep_info is not None and ep_info:
            episode.user_data["reward_greedy"].append(sum(ep_info['rewards_teams'][0].values()))
            episode.user_data["reward_coop"].append(sum(ep_info['rewards_teams'][1].values()))

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        episode.custom_metrics["reward_greedy"] = np.sum(episode.user_data["reward_greedy"])
        episode.custom_metrics["reward_coop"] = np.sum(episode.user_data["reward_coop"])

    '''
    def on_train_result(self, trainer, result, **kwargs):
        greedy_mse_fac = trainer.config['model']['custom_model_config']['greedy_mse_fac']
        if isinstance(greedy_mse_fac, list):
            s = PiecewiseSchedule(greedy_mse_fac[0], "torch", outside_value=greedy_mse_fac[1])
            trainer.workers.foreach_worker(
                lambda w: w.foreach_policy(
                    lambda p, p_id: p.model.update_config({'greedy_mse_fac': s(result['timesteps_total'])})))
    '''

def initialize():
    ray.init()
    register_env("coverage", lambda config: CoverageEnv(config))
    register_env("path_planning", lambda config: PathPlanningEnv(config))
    ModelCatalog.register_custom_model("adversarial", AdversarialModel)
    
def start_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("-o", "--override", help='Key in alternative_config from which to take data to override main config', default=None)
    parser.add_argument("-t", "--timesteps", help="Number of total time steps for training stop condition in millions", type=int, default=20)
    args = parser.parse_args()
    
    try:
        config_path = check_file(args.experiment)
    except FileNotFoundError:
        config_path = get_config_base() / (args.experiment + ".yaml")

    with open(config_path, "rb") as config_file:
        config = yaml.load(config_file)
    if args.override is not None:
        if not args.override in config['alternative_config']:
            print("Invalid alternative config key! Choose one from:")
            print(config['alternative_config'].keys())
            exit()
        update_dict(config, config['alternative_config'][args.override])
    config.pop('alternative_config', None)
    config['callbacks'] = EvaluationCallbacks

    initialize()        
    tune.run(
        MultiPPOTrainer,
        checkpoint_freq=10,
        stop={"timesteps_total": args.timesteps*1e6},
        keep_checkpoints_num=1,
        config=config,
        #local_dir="/tmp",
        trial_dirname_creator=trial_dirname_creator,
    )

def continue_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=dir_path)
    parser.add_argument("-t", "--timesteps", help="Number of total time steps for training stop condition in millions", type=int, default=20)
    parser.add_argument("-e", "--experiment", help="Path/id to training config", default=None)
    parser.add_argument("-o", "--override", help='Key in alternative_config from which to take data to override main config', default=None)

    args = parser.parse_args()
    
    with open(Path(args.checkpoint) / '..' / 'params.json', "rb") as config_file:
        config = json.load(config_file)
    
    if args.experiment is not None:
        try:
            config_path = check_file(args.experiment)
        except FileNotFoundError:
            config_path = get_config_base() / (args.experiment + ".yaml")
            
        with open(config_path, "rb") as config_file:
            update_dict(config, yaml.load(config_file)['alternative_config'][args.override])

    config['callbacks'] = EvaluationCallbacks

    checkpoint_file = Path(args.checkpoint) / ('checkpoint-' + os.path.basename(args.checkpoint).split('_')[-1])

    initialize()
    tune.run(
        MultiPPOTrainer,
        checkpoint_freq=20,
        stop={"timesteps_total": args.timesteps*1e6},
        restore=checkpoint_file,
        keep_checkpoints_num=1,
        config=config,
        #local_dir="/tmp",
        trial_dirname_creator=trial_dirname_creator,
    )

if __name__ == '__main__':
    start_experiment()
    exit()
    
    
    ### Cooperative
    run_experiment("./config/coverage.yaml", {"timesteps_total": 20e6}, None)
    run_experiment("./config/coverage_split.yaml", {"timesteps_total": 3e6}, None)
    run_experiment("./config/path_planning.yaml", {"timesteps_total": 20e6}, None)

    ### Adversarial
    continue_experiment("checkpoint_cov", {"timesteps_total": 60e6}, "./config/coverage.yaml", "adversarial")
    continue_experiment("checkpoint_split", {"timesteps_total": 20e6}, "./config/coverage_split.yaml", "adversarial")
    continue_experiment("checkpoint_flow", {"timesteps_total": 60e6}, "./config/path_planning.yaml", "adversarial")

    ### Re-adapt
    continue_experiment("checkpoint_cov_adv", {"timesteps_total": 90e6}, "./config/coverage.yaml", "cooperative")
    continue_experiment("checkpoint_split_adv", {"timesteps_total": 30e6}, "./config/coverage_split.yaml", "cooperative")
    continue_experiment("checkpoint_flow_adv", {"timesteps_total": 90e6}, "./config/path_planning.yaml", "cooperative")


