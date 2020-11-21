import numpy as np
import ray
import json
import os
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
#from model_team_adversarial import AdaptedVisionNetwork as AdversarialTeamModel
#from model_team_adversarial_2 import AdaptedVisionNetwork as AdversarialTeamModel2
from model_team_adversarial_2_vaegp import AdaptedVisionNetwork as AdversarialTeamModel2VAEGP
#from multiagent_ppo_trainer import MultiPPOTrainer
from multiagent_ppo_trainer_2 import MultiPPOTrainer as MultiPPOTrainer2
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import train_explainability as expl
import torch

mplstyle.use('fast')

from world import World
from world_super_2 import WorldSaliency as SuperWorld2
from world_teams import World as TeamWorld
#from world_teams_2 import WorldAdvDec as TeamWorld2
from world_teams_2 import World as TeamWorld2
from world_food import World as FoodWorld
from world_flow import World as FlowWorld
import pickle
from pathlib import Path

from guided_backprop import GuidedBackprop

import functools
class InterpretorDatasetCreator:
    def __init__(self, model):
        self.cnn_outputs = []
        self.gnn_outputs = []

        model.coop_convs[-1].register_forward_hook(self.record_cnn_output)
        model.greedy_convs[-1].register_forward_hook(self.record_cnn_output)
        model.GFL.register_forward_hook(self.record_gnn_output)

    def record_cnn_output(self, module, input_, output):
        self.cnn_outputs.append(output[0].detach().cpu().numpy())

    def record_gnn_output(self, module, input_, output):
        self.gnn_outputs.append(output[0].detach().cpu().numpy())

    def update_obs_and_clear(self, obs):
        for j in range(len(obs['agents'])):
            obs['agents'][j]['cnn_out'] = self.cnn_outputs[j]
            obs['agents'][j]['gnn_out'] = self.gnn_outputs[0][..., j]
        self.cnn_outputs, self.gnn_outputs = [], []
        return obs

class InterpretorInference:
    def __init__(self, model_checkpoint_path,
              seed=None):
        self.checkpoint_file = Path(model_checkpoint_path)
        with open(self.checkpoint_file.parent / 'config.json', 'r') as config_file:
            self.config = json.load(config_file)
        self.model = None

    def pred(self, dataset):
        if self.model is None:
            self.model = expl.load_model(self.checkpoint_file, expl.Model(dataset, self.config))

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )
        return np.array([self.model(x).detach().numpy()[0] for x, _ in loader])

import collections.abc

def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def fix_config(config):
    return config
    if 'operation_mode' not in config:
        print(config['env_config']['reward_type'])
        print(config['env_config']['episode_termination'])
        print(config['evaluation_config']['env_config']['reward_type'])
        print(config['evaluation_config']['env_config']['episode_termination'])
        exit()
        config['env_config']['operation_mode']

def serve(checkpoint_path):
    with open(checkpoint_path + '/../params.json') as json_file:
        checkpoint_config = fix_config(json.load(json_file))#['greedy'])
    #print(checkpoint_config)
    # checkpoint_config['env_config']['communication_range'] = 1
    # checkpoint_config['env_config']['agents']['visibility_distance'] = 0
    # checkpoint_config['env_config']['ensure_connectivity'] = False
    # checkpoint_config['env_config']['n_agents'] = 1
    # checkpoint_config['env_config']['n_obstacles'] = 30
    #checkpoint_config['env_config']['reward_type'] = "semi_cooperative"

    #env_config = checkpoint_config['env_config']
    env_config = update_dict(checkpoint_config['env_config'], checkpoint_config['evaluation_config']['env_config'])
    print(env_config)
    trainer_cfg = {
        "framework": "torch",
        "num_workers": 0,
        "seed": 0,
        "num_gpus": 0,
        "env_config": env_config,
        "model": checkpoint_config['model'],
    }
    #trainer_cfg['env_config']['reward_type'] = 'split_right_only'
    #trainer_cfg['env_config']['reward_type'] = 'local'
    #trainer_cfg['env_config']['episode_termination'] = 'early_coop_right'

    #print(trainer_cfg['env_config']['disabled_teams_step'])
    #trainer_cfg['env_config']['disabled_teams_step'] = [False, False]
    #trainer_cfg['env_config']['disabled_teams_comms'] = [True, False]
    #trainer_cfg['env_config']['n_agents'] = [1, 1]
    #trainer_cfg['env_config']['communication_range'] = 32
    #trainer_cfg['model']['custom_model_config']['filter_adv'] = True
    #trainer_cfg['model']['custom_model_config']['agent_split'] = 0
    #trainer_cfg['model']['custom_model_config']['agent_noise'] = 1
    #trainer_cfg['model']['custom_model_config']['agent_noise_stddev'] = 100
    #trainer_cfg['model']['custom_model_config']['coop_gpvae_pre_train_path'] = "./gpvae_16_fine_tune_sch_kldw_0.050_cont_0.005.pth"
    #trainer_cfg['model']['custom_model_config']['train_coop_gpvae'] = True # for guided backprop of all agents
    #trainer_cfg['model']['custom_model_config']['filter_adv'] = True
    #trainer_cfg['model']['custom_model_config']['graph_aggregation'] = "median"
    #trainer_cfg['model']['custom_model_config']['filter_zero_only'] = False
    filter_biases = [
        [43, -3], # 0.9
        [42, -5], # 0.8
        [40, -6], # 0.7
        [40, -7], # 0.6
        [37, -7], # 0.5
        [36, -8], # 0.4
        [34, -8], # 0.3
        [33, -9], # 0.2
    ]
    idx = 0
    #trainer_cfg['model']['custom_model_config']['filter_outlier_bias'] = filter_biases[idx][0]
    #trainer_cfg['model']['custom_model_config']['filter_indep_bias'] = filter_biases[idx][1]
    #trainer_cfg['env_config']['n_agents'] = [1, 3]
    #trainer_cfg['env_config']['world_shape'] = [48, 48]
    #trainer_cfg['env_config']['map_mode'] = "random_teams_far"
    #trainer_cfg['env_config']['communication_range'] = 16

    trainer = MultiPPOTrainer2(
        env=checkpoint_config['env'],
        config=trainer_cfg
    )
    checkpoint_file = checkpoint_path + '/checkpoint-' + os.path.basename(checkpoint_path).split('_')[-1]
    trainer.restore(checkpoint_file)

    envs = {
        'foodworld': FoodWorld,
        'flowworld': FlowWorld,
        'teamworld2': TeamWorld2
    }
    env = envs[checkpoint_config['env']](checkpoint_config['env_config'])
    env.seed(1)
    #plt.ion()
    obs = env.reset()
    rewards = 0
    states = trainer.get_policy().get_initial_state()

    best_perfs = []
    rewards = {0: [0], 1: [0], 's': [0]}
    mean_rewards = {0: [], 1: [], 'cov': []}

    #interpr = InterpretorInference("./results/0721/explainability_checkpoints/explainability_cov_cov_only_local_57_hu8xcpq_1560/best_model_270_val_ap=0.8345176157248761.pth")
    #dset_creator = InterpretorDatasetCreator(trainer.get_policy().model)

    model = trainer.get_policy().model

    frame_index = 0
    all_rew = 0
    weights = []
    logvars = []
    for i in range(100):
        while True:
            #weights = model.coop_filter_adv(obs)
            #obs['gso'] *= weights[0].detach().numpy()
            #print(obs['gso'])
            #env.render()

            actions = trainer.compute_action(obs)
            #weights.append(model.weight.numpy())
            #print(np.array(weights).mean(axis=0))
            #logvars += model.logvars[1:]

            #from sim_heuristics_random import heuristic
            #for i in range(1, len(obs['agents'])):
            #    actions[i] = heuristic(env, obs['agents'][i]).value

            #obs = dset_creator.update_obs_and_clear(obs)
            #interpr_pred = interpr.pred(expl.CoverageDataset(obs, interpr.config))

            #b = GuidedBackprop(trainer.get_policy().model)
            #b.calculate_gradients({"obs": obs}, 1, guided=True)
            obs, reward, done, info = env.step(actions)

            #for i in range(10):
            #print(np.mean(b.gradients[0]))
            #greedy_enc, _, _ = model.greedy_convs.encode(torch.Tensor([obs['agents'][0]['map']]).permute(0, 3, 1, 2))
            #greedy_enc, _, _ = model.coop_vaegp.vae.encode(torch.Tensor([obs['agents'][0]['map']]).permute(0, 3, 1, 2))
            #_, greedy_dec, _ = model.coop_vaegp.vae.decode(greedy_enc)
            f = env.render() #b.gradients) #obs_to_abs(greedy_dec.detach(), obs['agents'][0]['pos']))
            #f = env.render(interpr_pred[0], stepsize=1.0) #saliency_obs=b.gradients) #, interpreter_obs=p[0])
            #f.savefig(f"{checkpoint_path}/rendering/{frame_index:05d}.png", dpi=200)
            frame_index += 1
            #if frame_index == 2000:
            #    return
            #print(info)
            for j in range(2):
                rewards[j][-1] += np.mean(list(info['rewards_teams'][j].values()))
            #print(rewards[0][-1], rewards[1][-1])
            rewards['s'][-1] += reward/16
            #print(info)
            #print(rewards)
            # input("next")
            # if i == 200:
            #    exit()
            #print(info)
            #if env.timestep == 3: #int(1.5*(info['coverable_area']/checkpoint_config['env_config']['n_agents'][1])):
            #if env.timestep == 104:
            if done: #int(info['coverable_area']):
                #f = env.render() #saliency_obs=b.gradients)#, saliency_pos=b.gradients_gnn[0])
                #torch.set_printoptions(precision=2, sci_mode=False)

                if False: #i == 3:
                    all_w = np.array(weights)[:, 0].flatten()
                    #all_w = np.array(weights)[:, 1:].flatten()
                    plt.hist(all_w, bins=100)
                    print(np.mean(all_w))
                    plt.show()
                    exit()

                    plt.hist(torch.stack(logvars).flatten().numpy(), bins=100)
                    plt.show()

                for j in range(2):
                    print("R", i, j, np.mean(rewards[j]), np.std(rewards[j]))
                    rewards[j].append(0)
                print(np.mean(rewards['s']))
                rewards['s'].append(0)
                #input("Done")

                # best_perfs.append(info['current_global_coverage'])
                # print(np.mean(best_perfs), env.timestep)
                #if len(best_perfs) == 50:
                #    exit()

                # input("continue")
                obs = env.reset()
                break


if __name__ == "__main__":
    ray.init()
    #ModelCatalog.register_custom_model("vis_torch_adv_team", AdversarialTeamModel)
    #ModelCatalog.register_custom_model("vis_torch_adv_team_2", AdversarialTeamModel2)
    ModelCatalog.register_custom_model("vis_torch_adv_team_2_vaegp", AdversarialTeamModel2VAEGP)

    register_env("teamworld2", lambda config: TeamWorld2(config))
    register_env("foodworld", lambda config: FoodWorld(config))
    register_env("flowworld", lambda config: FlowWorld(config))
    try:
        #serve("./results/1009/MultiPPO_teamworld2_13a0a_00000/checkpoint_1880") # naive adversarial (T w/o)
        #serve("./results/1009/Multi_DDPPO_teamworld2_b7839_00000/checkpoint_1000") # naive adversarial (T w/o)
        #serve("./results/0922/MultiPPO_teamworld2_c8d29_00000/checkpoint_410") # coop
        #serve("./results/1009/MultiPPO_teamworld2_351c7_00000/checkpoint_1600") # T M

        #serve("./results/1028/MultiPPO_teamworld2_c80b2_00000/checkpoint_440") # no coll, empty
        #serve("./results/1028/MultiPPO_teamworld2_d7f8b_00000/checkpoint_440") # coll, empty
        serve("./results/1028/MultiPPO_teamworld2_8be2d_00000/checkpoint_440") # coll, 0.8
    except KeyboardInterrupt:
        ray.shutdown()
        pass
