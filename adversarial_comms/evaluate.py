import argparse
import collections.abc
import json
import numpy as np
import os
import pandas as pd
import ray
import time
import traceback

from pathlib import Path
from ray.rllib.models import ModelCatalog
from ray.tune.logger import NoopLogger
from ray.tune.registry import register_env
from ray.util.multiprocessing import Pool

from .environments.coverage import CoverageEnv
from .environments.path_planning import PathPlanningEnv
from .models.adversarial import AdversarialModel
from .trainers.multiagent_ppo import MultiPPOTrainer

def heuristic(env, state):
    state_obstacles, state_coverage = (state['map'][...,i] for i in range(2))
    half_state_shape = (np.array(state_obstacles.shape)/2).astype(int)
    actions_deltas = {
        world.Action.MOVE_RIGHT:  [ 0,  1],
        world.Action.MOVE_LEFT:   [ 0, -1],
        world.Action.MOVE_UP:     [-1,  0],
        world.Action.MOVE_DOWN:   [ 1,  0],
        #world.Action.NOP:         [ 0,  0]
    }

    options_free = []
    options_uncovered = []
    for a, dp in actions_deltas.items():
        p = half_state_shape + dp
        if state_obstacles[p[world.Y], p[world.X]] > 0:
            continue
        options_free.append(a)

        if state_coverage[p[world.Y], p[world.X]] > 0:
            continue
        options_uncovered.append(a)

    if len(options_uncovered) > 0:
        return random.choice(options_uncovered)
    elif len(options_free) > 0:
        return random.choice(options_free)
    return world.Action.NOP

def run_random(checkpoint_path, trial, ep_len, cfg_update, comm_index):
    try:
        t0 = time.time()
        with open(checkpoint_path + '/../params.json') as json_file:
            cfg = json.load(json_file)
            if 'greedy' in cfg:
                cfg = cfg['greedy']

        cfg = update_dict(cfg, cfg_update)
        if 'evaluation_config' in cfg:
            cfg['env_config'] = update_dict(cfg['env_config'], cfg['evaluation_config']['env_config'])
        envs = {'coverage': CoverageEnv, 'path_planning': PathPlanningEnv}
        env = envs[cfg['env']](cfg['env_config'])
        env.seed(trial) # same environments for all distances
        obs = env.reset()
        #states = trainer.get_policy().get_initial_state()

        results = []

        for i in range(ep_len):
            actions = tuple([heuristic(env, state) for state in obs['agents']])
            obs, reward, done, info = env.step(actions)
            results.append(list(info['rewards'].values()))
        print("Done", time.time() - t0)
    except Exception as e:
        print(e, traceback.format_exc())
        raise
    results = np.array(results)
    print(results)
    return checkpoint_path, trial, comm_index, results

def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def run_trial(checkpoint_path, trial, cfg_update):
    try:
        t0 = time.time()
        with open(checkpoint_path + '/../params.json') as json_file:
            cfg = json.load(json_file)
            if 'greedy' in cfg:
                cfg = cfg['greedy']

        checkpoint_file = checkpoint_path + '/checkpoint-' + os.path.basename(checkpoint_path).split('_')[-1]

        env_config = cfg['env_config'] if 'evaluation_config' not in cfg else update_dict(cfg['env_config'], cfg['evaluation_config']['env_config'])

        trainer_cfg = update_dict({
            "framework": "torch",
            "seed": trial,
            "num_workers": 0,
            "env_config": cfg['env_config'],
            "model": cfg['model']
        }, cfg_update)

        trainer = MultiPPOTrainer(
            env=cfg['env'],
            logger_creator=lambda config: NoopLogger(config, ""),
            config=trainer_cfg
        )
        trainer.restore(checkpoint_file)

        envs = {'coverage': CoverageEnv, 'path_planning': PathPlanningEnv}
        env = envs[cfg['env']](cfg['env_config'])
        env.seed(trial)
        obs = env.reset()

        results = []
        for i in range(env_config['max_episode_len']):
            actions = trainer.compute_action(obs)
            obs, reward, done, info = env.step(actions)
            results.append({
                'trial': trial,
                'rewards': list(info['rewards'].values())
            })

        print("Done", time.time() - t0)
    except Exception as e:
        print(e, traceback.format_exc())
        raise
    df = pd.DataFrame(results)
    return df

def path_to_hash(path):
    path_split = path.split('/')
    checkpoint_number_string = path_split[-1].split('_')[-1]
    path_hash = path_split[-2].split('_')[-2]
    return path_hash + '-' + checkpoint_number_string

def serve_config(checkpoint_path, trials, cfg_change={}, run_function=run_trial):
    with Pool() as p:
        results = pd.concat(p.starmap(run_function, [(checkpoint_path, t, cfg_change) for t in range(trials)]))
    return results

def initialize():
    ray.init()
    register_env("coverage", lambda config: CoverageEnv(config))
    register_env("path_planning", lambda config: PathPlanningEnv(config))
    ModelCatalog.register_custom_model("adversarial", AdversarialModel)

def trial_nocomm(env_config_func, prefix):
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out_path")
    parser.add_argument("-t", "--trials", type=int, default=100)
    args = parser.parse_args()

    initialize()
    results = []
    for comm in [False, True]:
        cfg_change={'env_config': env_config_func(comm)}
        df = serve_config(args.checkpoint, args.trials, cfg_change=cfg_change)
        df['comm'] = comm
        results.append(df)

    filename = prefix + "-" + path_to_hash(args.checkpoint) + ".pkl"
    pd.concat(results).to_pickle(Path(args.out_path)/filename)

def trial_nocomm_coop():
    # Cooperative agents can communicate or not (without comm interference from adversarial agent)
    trial_nocomm(lambda comm: {
        'disabled_teams_comms': [True, not comm],
        'disabled_teams_step': [True, False]
    }, "eval_coop")

def trial_nocomm_adv():
    # all cooperative agents can still communicate, but adversarial communication is switched
    trial_nocomm(lambda comm: {
        'disabled_teams_comms': [not comm, False], # en/disable comms for adv and always enabled for coop
        'disabled_teams_step': [False, False] # both teams operating
    }, "eval_adv")

if __name__ == "__main__":
    trial_coop_nocomm()

