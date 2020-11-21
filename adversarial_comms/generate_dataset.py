import numpy as np
import ray
from ray.util.multiprocessing import Pool
import json
import os
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.tune.logger import NoopLogger
#from model_team_adversarial import AdaptedVisionNetwork as AdversarialTeamModel
#from model_team_adversarial_2 import AdaptedVisionNetwork as AdversarialTeamModel2
from model_team_adversarial_2_vaegp import AdaptedVisionNetwork as AdversarialTeamModel2VAEGP
from multiagent_ppo_trainer_2 import MultiPPOTrainer as MultiPPOTrainer2
import matplotlib.style as mplstyle
mplstyle.use('fast')

from world_teams_2 import World as TeamWorld2
from world_flow import WorldOverview as FlowWorld
import pickle
import torch

import copy

def generate(seed, checkpoint_path, sample_iterations, termination_mode, frame_take_prob=0.1, disable_adv_comm=False, ensure_conn=False, t_fac=1.5):
    with open(checkpoint_path + '/../params.json') as json_file:
        checkpoint_config = json.load(json_file)

    checkpoint_config['env_config']['ensure_connectivity'] = ensure_conn

    checkpoint_config['env_config']['disabled_teams_comms'] = [disable_adv_comm, False]
    checkpoint_config['env_config']['disabled_teams_step'] = [False, False]

    trainer_cfg = {
        "framework": "torch",
        "num_workers": 1,
        "num_gpus": 1,
        "env_config": checkpoint_config['env_config'],
        "model": checkpoint_config['model'],
        "seed": seed
    }

    trainer = MultiPPOTrainer2(
        logger_creator=lambda config: NoopLogger(config, ""),
        env=checkpoint_config['env'],
        config=trainer_cfg
    )
    checkpoint_file = checkpoint_path + '/checkpoint-' + os.path.basename(checkpoint_path).split('_')[-1]
    trainer.restore(checkpoint_file)

    envs = {
        'flowworld': FlowWorld,
        'teamworld2': TeamWorld2
    }
    env = envs[checkpoint_config['env']](checkpoint_config['env_config'])
    env.seed(seed)
    obs = env.reset()

    samples = []
    model = trainer.get_policy().model

    cnn_outputs = []
    def record_cnn_output(module, input_, output):
        cnn_outputs.append(output[0].detach().cpu().numpy())
    gnn_outputs = []
    def record_gnn_output(module, input_, output):
        gnn_outputs.append(output[0].detach().cpu().numpy())
    #model.coop_convs[-1].register_forward_hook(record_cnn_output)
    #model.greedy_convs[-1].register_forward_hook(record_cnn_output)
    model.GFL.register_forward_hook(record_gnn_output)

    while len(samples) < sample_iterations:
        actions = trainer.compute_action(obs)
        for j in range(1, sum(checkpoint_config['env_config']['n_agents'])):
            #obs['agents'][j]['cnn_out'] = cnn_outputs[j]
            z, mu, log = model.coop_vaegp.vae.encode(torch.from_numpy(np.array([obs['agents'][j]['map']])).float().permute(0,3,1,2))
            obs['agents'][j]['cnn_out'] = z[0].detach()
            obs['agents'][j]['gnn_out'] = gnn_outputs[0][..., j]
        cnn_outputs = []
        gnn_outputs = []

        if np.random.rand() <= frame_take_prob:
            samples.append(copy.deepcopy({'obs': obs, 'actions': actions}))
            print(len(samples))

        obs, reward, done, info = env.step(actions)
        if (termination_mode == 'path' and done) or (termination_mode == 'cov' and env.timestep == int(t_fac*(info['coverable_area']/checkpoint_config['env_config']['n_agents'][1]))):
            obs = env.reset()
    return samples

def run(seed, checkpoint_path, samples, workers, generated_path, termination_mode, frame_take_prob=0.2, disable_adv_comm=False, t_fac=1.5):
    results = []
    with Pool(workers) as p:
       for res in p.starmap(generate, [(seed+i, checkpoint_path, int(samples/workers), termination_mode, frame_take_prob, disable_adv_comm, t_fac) for i in range(workers)]):
            results += res
    print("DONE", len(results))
    pickle.dump(results, open(generated_path, "wb"))

if __name__ == "__main__":
    ray.init()
    #ModelCatalog.register_custom_model("vis_torch_adv_team", AdversarialTeamModel)
    #ModelCatalog.register_custom_model("vis_torch_adv_team_2", AdversarialTeamModel2)
    ModelCatalog.register_custom_model("vis_torch_adv_team_2_vaegp", AdversarialTeamModel2VAEGP)

    register_env("teamworld2", lambda config: TeamWorld2(config))
    register_env("flowworld", lambda config: FlowWorld(config))

    # cooperative trainings 
    #checkpoint_path = "/local/scratch/jb2270/corl_evaluation/MultiPPO/MultiPPO_teamworld2_0_2020-07-19_01-15-57_hu8xcpq/checkpoint_1560" # coverage
    #checkpoint_path = "/local/scratch/jb2270/corl_evaluation/MultiPPO/MultiPPO_teamworld2_0_2020-07-18_23-44-12k2_enqa8/checkpoint_150" # split
    #checkpoint_path = "/local/scratch/jb2270/corl_evaluation/MultiPPO/MultiPPO_flowworld_0_2020-07-16_00-47-53k6vmhzpl/checkpoint_1300" # flow 7x7
    #checkpoint_path = "/local/scratch/jb2270/corl_evaluation/MultiPPO/MultiPPO_flowworld_0_2020-07-16_10-53-29pe06c7bw/checkpoint_3100" # flow 24x24

    # adversarial
    #checkpoint_path = "/local/scratch/jb2270/corl_evaluation/MultiPPO/MultiPPO_teamworld2_0_2020-07-20_23-31-52zj__fmp3/checkpoint_4600" # cov
    #checkpoint_path = "/local/scratch/jb2270/corl_evaluation/MultiPPO/MultiPPO_teamworld2_0_2020-07-19_09-34-02u_h77o5y/checkpoint_1400" # split
    #checkpoint_path = "/local/scratch/jb2270/corl_evaluation/MultiPPO/MultiPPO_flowworld_0_2020-07-16_10-59-27c8iboc7_/checkpoint_3800" # flow 7x7
    #checkpoint_path = "/local/scratch/jb2270/corl_evaluation/MultiPPO/MultiPPO_flowworld_0_2020-07-24_00-56-5896e2idut/checkpoint_8100" # flow 24x24

    #re-adapt
    #checkpoint_path = "/local/scratch/jb2270/corl_evaluation/MultiPPO/MultiPPO_teamworld2_0_2020-07-27_00-38-42vpz3xf0k/checkpoint_5690" # cov
    #checkpoint_path = "/local/scratch/jb2270/corl_evaluation/MultiPPO/MultiPPO_teamworld2_0_2020-07-27_00-48-12zecm5uk7/checkpoint_2190" # split
    
    #checkpoint_path = "/local/scratch/jb2270/vaegp_eval/MultiPPO/MultiPPO_teamworld2_0_2020-08-23_11-27-05u14jlcjb/checkpoint_1560" # simple
    
    #checkpoint_path = "/local/scratch/jb2270/vaegp_eval/MultiPPO/MultiPPO_teamworld2_0_2020-08-24_20-43-40mnea2uga/checkpoint_1560" # train with frozen VAE
    checkpoint_path = "/local/scratch/jb2270/vaegp_eval/MultiPPO/MultiPPO_teamworld2_c8d29_00000/checkpoint_410"

    termination_mode = "cov" # cov/path
    
    checkpoint_num = checkpoint_path.split("_")[-1]
    checkpoint_id = checkpoint_path.split("/")[-2].split("-")[-1]
    #generate(0, checkpoint_path, 1000, 0.1, 1.5)
    #exit()
    run(0, checkpoint_path, 50000, 32, f"/local/scratch/jb2270/datasets_corl/explainability_data_{checkpoint_id}_{checkpoint_num}_train.pkl",termination_mode, disable_adv_comm=True)
    run(1, checkpoint_path, 10000, 32, f"/local/scratch/jb2270/datasets_corl/explainability_data_{checkpoint_id}_{checkpoint_num}_valid.pkl", termination_mode, disable_adv_comm=True)
    run(2, checkpoint_path, 1000, 1, f"/local/scratch/jb2270/datasets_corl/explainability_data_{checkpoint_id}_{checkpoint_num}_test.pkl", termination_mode, disable_adv_comm=True, frame_take_prob=1.0, t_fac=4)
    #run(2, checkpoint_path, 1000, 1, f"/local/scratch/jb2270/datasets_corl/explainability_data_{checkpoint_id}_{checkpoint_num}_nocomm_test.pkl", termination_mode, disable_adv_comm=True, frame_take_prob=1.0)

