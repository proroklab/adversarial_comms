from abc import ABC

import ray
import random

import numpy as np

from enum import Enum
from gym import spaces
from ray.rllib import Policy
from ray.rllib.agents import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ParallelRollouts, SelectExperiences
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.utils import override
from ray.rllib.utils.typing import TrainerConfigDict
from ray.util.iter import LocalIterator
from ray.tune.registry import register_env

DEFAULT_CONFIG = with_common_config({})

class Action(Enum):
    NOP         = 0
    MOVE_RIGHT  = 1
    MOVE_LEFT   = 2
    MOVE_UP     = 3
    MOVE_DOWN   = 4

X = 1
Y = 0

class RandomHeuristicPolicy(Policy, ABC):
    """
    Based on
    https://github.com/ray-project/ray/blob/releases/1.0.1/rllib/examples/policy/random_policy.py
    Visit a random uncovered neighboring cell or a random cell if all are covered
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def single_random_heuristic(self, obs):
        state_obstacles, state_coverage = (obs[:, :, i] for i in range(2))
        half_state_shape = (np.array(state_obstacles.shape)/2).astype(int)
        actions_deltas = {
            Action.MOVE_RIGHT.value:  [ 0,  1],
            Action.MOVE_LEFT.value:   [ 0, -1],
            Action.MOVE_UP.value:     [-1,  0],
            Action.MOVE_DOWN.value:   [ 1,  0],
        }

        options_free = []
        options_uncovered = []
        for a, dp in actions_deltas.items():
            p = half_state_shape + dp
            if state_obstacles[p[Y], p[X]] > 0:
                continue
            options_free.append(a)

            if state_coverage[p[Y], p[X]] > 0:
                continue
            options_uncovered.append(a)

        if len(options_uncovered) > 0:
            return random.choice(options_uncovered)
        elif len(options_free) > 0:
            return random.choice(options_free)
        return NOP.value

    @override(Policy)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):

        obs_batch = restore_original_dimensions(
            np.array(obs_batch, dtype=np.float32),
            self.observation_space,
            tensorlib=np)

        r = np.array([[self.single_random_heuristic(map_batch) for map_batch in agent['map']] for agent in obs_batch['agents']])
        return r.transpose(), [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


def execution_plan(workers: WorkerSet,
                   config: TrainerConfigDict) -> LocalIterator[dict]:
    rollouts = ParallelRollouts(workers, mode="async")

    # Collect batches for the trainable policies.
    rollouts = rollouts.for_each(
        SelectExperiences(workers.trainable_policies()))

    # Return training metrics.
    return StandardMetricsReporting(rollouts, workers, config)


RandomHeuristicTrainer = build_trainer(
    name="RandomHeuristic",
    default_config=DEFAULT_CONFIG,
    default_policy=RandomHeuristicPolicy,
    execution_plan=execution_plan)
