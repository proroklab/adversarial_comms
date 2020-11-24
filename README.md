# Adversarial Comms
Code accompanying the paper
> [The Emergence of Adversarial Communication in Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2008.02616)\
> Jan Blumenkamp, Amanda Prorok\
> (University of Cambridge)\
> _arXiv: 2008.02616_.

[![Video preview](https://img.youtube.com/vi/o1Nq9XoSU6U/0.jpg)](https://www.youtube.com/watch?v=o1Nq9XoSU6U)

## Installation
Clone the repository, change directory into its root and run:
```
pip install -e .
```
This will install the package and all requirements. It will also set up the entry points we are referring to later in these instructions.

## Training
Generally, training is performed for the policies and for the interpreter. We first explain the three policy training steps (cooperative, self-interested, and re-adaptation) for all three experiments (coverage, split coverage and path planning) and then for the interpreters.

The policy training follows this scheme:
```
train_policy [experiment] -t [total time steps in millions]
continue_policy [cooperative checkpoint path] -t [total time steps] -e [experiment] -o self_interested
continue_policy [self-interested checkpoint path] -t [total time steps] -e [experiment] -o re_adapt
```
where `experiment` is one of `{coverage, coverage_split, path_planning}`, `-t` is the total number of time steps at which the experiment is to be terminated (note that this is not per call, but total time steps, so if a policy is trained with `train_policy -t 20` and then continued with `continue_policy -t 20` it will terminate immediately) and `-o` is a config option (one of `{self_interested, re_adapt}` as can be found in the `alternative_config` key in each of the config files in `config`).

When running each experiment, Ray will print the trial name to the terminal, which looks something like `MultiPPO_coverage_f4dc4_00000`. By default, Ray will create the directory `~/ray_results/MultiPPO` in which the trial with the given name can be found with its checkpoint. `continue_policy` expects the path to one of such checkpoints, for example `~/ray_results/MultiPPO/MultiPPO_coverage_f4dc4_00000/checkpoint_440`. The first `continue_policy` expects the checkpoint generated in the first `train_policy` call and the second `continue_policy` the checkpoint generated in the first `continue_policy` call. You should take note of each experiment's checkpoint path.

### Standard Coverage
```
train_policy coverage -t 20
continue_policy [cooperative checkpoint path] -t 60 -e coverage -o self_interested
continue_policy [adversarial checkpoint path] -t 80 -e coverage -o re_adapt
```

### Split coverage
```
train_policy coverage_split -t 3
continue_policy [cooperative checkpoint path] -t 20 -e coverage_split -o self_interested
continue_policy [adversarial checkpoint path] -t 30 -e coverage_split -o re_adapt
```

### Path Planning
```
train_policy path_planning -t 20
continue_policy [cooperative checkpoint path] -t 60 -e path_planning -o self_interested
continue_policy [adversarial checkpoint path] -t 80 -e path_planning -o re_adapt
```

## Evaluation
We provide three methods for evaluation:

1) `evaluate_coop`: Evaluate cooperative only performance while disabling self-interested agents with and without communication among cooperative agents.
2) `evaluate_adv`: Evaluate cooperative and self-interested agents with and without communication between cooperative and self-interested agents (cooperative agents can always communicate to each other).
3) `evaluate_random`: Run a random policy that visits random neighboring (preferably uncovered) cells.

The evaluation is run as
```
evaluate_{coop, adv} [checkpoint path] [result path] --trials 100
evaluate_random [result path] --trials 100
```
for 100 evaluation runs with different seeds. The resulting file is a Pandas dataframe containing the rewards for all agents at every time step. It can be processed and visualized by running `evaluate_plot [pickled data path]`.

Additionally, a checkpoint can be rolled out and rendered for a randomly generated environment with `evaluate_serve [checkpoint_path] --seed 0`. 

## Citation
If you use any part of this code in your research, please cite our paper:
```
@article{blumenkamp2020adversarial,
  title={The Emergence of Adversarial Communication in Multi-Agent Reinforcement Learning},
  author={Blumenkamp, Jan and Prorok, Amanda},
  journal={Conference on Robot Learning (CoRL)},
  year={2020}
}
```
