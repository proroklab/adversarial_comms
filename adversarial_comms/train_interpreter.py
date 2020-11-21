import pickle
import torch

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

import utils.graphML as gml
import utils.graphTools
import numpy as np

torch, nn = try_import_torch()
from torch.utils.data import Dataset

from torch.optim import SGD, Adam
import torch.nn.functional as F

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Precision, Recall, Fbeta, Loss, RunningAverage
#from ignite.contrib.metrics import ROC_AUC, AveragePrecision
from ignite.handlers import ModelCheckpoint, global_step_from_engine, EarlyStopping, TerminateOnNan
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *

from torchsummary import summary

from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import json
import random
from pathlib import Path
import time
import os
import copy

# https://ray.readthedocs.io/en/latest/using-ray-with-pytorch.html

X = 1
Y = 0

def get_transpose_cnn(inp_features, out_shape, out_classes):
    return [
        nn.ConvTranspose2d(in_channels=inp_features, out_channels=64, kernel_size=3, stride=1),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2),
        nn.LeakyReLU(inplace=True),
        nn.ZeroPad2d([1,1,1,1]),
        nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2),
        nn.LeakyReLU(inplace=True),
        nn.ZeroPad2d([1,1,1,1]),
        nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(8, out_classes, 3, 1),
        nn.Sigmoid(),
    ]

def get_upsampling_cnn(inp_features, out_shape, out_classes):
    if out_shape == 7:
        return [
            nn.ZeroPad2d([2]*4),
            nn.Conv2d(in_channels=inp_features, out_channels=16, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d([1]*4),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=4),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=8, out_channels=out_classes, kernel_size=4),
            nn.Sigmoid(),
        ]
    elif out_shape == 12:
        return [
            nn.ZeroPad2d([2]*4),
            nn.Conv2d(in_channels=inp_features, out_channels=16, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d([1]*4),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=4),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=8, out_channels=out_classes, kernel_size=3),
            nn.Sigmoid(),
        ]
    elif out_shape == 24:
        return [
            nn.ZeroPad2d([2]*4),
            nn.Conv2d(in_channels=inp_features, out_channels=32, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d([1]*4),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d([1]*4),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d([1]*4),
            nn.Conv2d(in_channels=8, out_channels=out_classes, kernel_size=3),
            nn.Sigmoid(),
        ]
    elif out_shape == 48:
        return [
            nn.ZeroPad2d([2]*4),
            nn.Conv2d(in_channels=inp_features, out_channels=64, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d([1]*4),
            nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d([1]*4),
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d([1]*4),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d([1]*4),
            nn.Conv2d(in_channels=16, out_channels=out_classes, kernel_size=3),
            nn.Sigmoid()
        ]
    assert False

class Model(nn.Module):
    def __init__(self, dataset, config):
        nn.Module.__init__(self)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.inp_features = {
            'gnn': dataset.gnn_features,
            'cnn': dataset.cnn_features,
            'gnn_cnn': dataset.cnn_features+dataset.gnn_features
        }[self.config['nn_mode']]

        if self.config['format']=='relative':
            if self.config['pred_mode'] == 'global':
                self.out_size = dataset.world_shape[0]*2
            elif self.config['pred_mode'] == 'local':
                self.out_size = dataset.obs_size
            else:
                raise NotImplementedError
        elif self.config['format']=='absolute':
            self.out_size = dataset.world_shape[0]
        else:
            raise NotImplementedError

        if self.config['type'] == 'cov':
            self.classes = 1 if self.config['prediction']=='cov_only' else 2
        elif self.config['type'] == 'path':
            self.classes = 3 if self.config['prediction']=='all' else 1
        else:
            raise NotImplementedError("Invalid type")

        layers = get_upsampling_cnn(self.inp_features, self.out_size, self.classes)
        cnn = nn.Sequential(*layers)
        #summary(cnn, device="cpu", input_size=(self.inp_features, 1, 1))
        self._post_cnn = cnn.to(self.device)

    @override(ModelV2)
    def forward(self, input_dict):
        agent_observations = input_dict["obs"]['agents']
        batch_size = input_dict["obs"]['gso'].shape[0]

        prediction = torch.empty(batch_size, len(agent_observations), self.classes, self.out_size, self.out_size).to(
            self.device)
        for this_id, this_state in enumerate(agent_observations):
            gnn_out = this_state['gnn_out']
            cnn_out = this_state['cnn_out']
            this_entity = {
                'gnn': gnn_out,
                'cnn': cnn_out,
                'gnn_cnn': torch.cat([gnn_out, cnn_out], dim=1)
            }[self.config['nn_mode']]
            prediction[:, this_id] = self._post_cnn(this_entity.view(batch_size, self.inp_features, 1, 1))

        return prediction.double()

class BaseDataset(Dataset):
    def __init__(self, path):
        try:
            with open(Path(path), "rb") as f:
                self.data = pickle.load(f)
                assert (len(self.data) > 0)
        except TypeError:
            self.data = [{'obs': path}]
        self.world_shape = self.data[0]['obs']['state'].shape[:2]
        self.obs_size = self.data[0]['obs']['agents'][0]['map'].shape[0]
        self.cnn_features = self.data[0]['obs']['agents'][-1]['cnn_out'].shape[0]
        self.gnn_features = self.data[0]['obs']['agents'][-1]['gnn_out'].shape[0]

    def __len__(self):
        return len(self.data)

    def get_coverable_area(self, idx):
        coverable_area = ~(self.data[idx]['obs']['state'][...,0] > 0)
        return np.sum(coverable_area)

    def get_coverage_fraction(self, idx):
        coverable_area = ~(self.data[idx]['obs']['state'][...,0] > 0)
        covered_area = self.data[idx]['obs']['state'][...,1] & coverable_area
        return np.sum(covered_area) / np.sum(coverable_area)

    def to_agent_coord_frame(self, m, state_size, pose, fill=0):
        half_state_shape = np.array([state_size / 2] * 2, dtype=np.int)
        padded = np.pad(m, ([half_state_shape[Y]] * 2, [half_state_shape[X]] * 2), mode='constant',
                        constant_values=fill)
        return padded[pose[Y]:pose[Y] + state_size, pose[X]:pose[X] + state_size]

class CoverageDataset(BaseDataset):
    def __init__(self, path, config):
        # is_relative: Agent relative or world absolute prediction
        # cov_only: Predict only coverage or predict both coverage and map
        # is_global: Predict local coverage and map or global coverage and map

        super().__init__(path)
        self.is_relative=config['format']=='relative'
        self.cov_only=config['prediction']=='cov_only'
        self.is_global=config['pred_mode']=='global'
        self.skip_agents=config['skip_agents'] if 'skip_agents' in config else 0
        self.stop_agents=config['stop_agents'] if 'stop_agents' in config else None

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        y = []
        weights = []
        for agent_obs in self.data[idx]['obs']['agents'][self.skip_agents:self.stop_agents]:
            if self.is_global:
                obs_cov = self.data[idx]['obs']['state'][...,1]
                obs_map = self.data[idx]['obs']['state'][...,0]
                if self.is_relative:
                    obs_cov = self.to_agent_coord_frame(obs_cov, self.obs_size, agent_obs['pos'], fill=0)
                    obs_map = self.to_agent_coord_frame(obs_map, self.obs_size, agent_obs['pos'], fill=1)
            else:
                if self.is_relative:
                    # directly use agent's relative view coverage
                    obs_cov = agent_obs['map'][..., 1]
                    obs_map = agent_obs['map'][..., 0]
                else:
                    # shift the agent's local coverage to an absolute view
                    m = np.roll(agent_obs['map'], agent_obs['pos'], axis=(0,1))[int(self.obs_size/2):,int(self.obs_size/2):]
                    obs_cov = m[...,1]
                    obs_map = m[...,0]

            if self.cov_only:
                # only predict local coverage and use world map as mask
                y.append([obs_cov])
                weights.append([(~obs_map.astype(np.bool)).astype(np.int)])
            else:
                # predict both local coverage and world map, but mask out everything outside the world shifted to the agents position
                d = np.stack([obs_cov, obs_map], axis=0)
                y.append(d)
                weight = np.ones(obs_cov.shape)
                if self.is_relative:
                    weight = self.to_agent_coord_frame(weight, self.obs_size, agent_obs['pos'], fill=0)
                weight = np.stack([weight]*2, axis=0)
                #print(d.shape, weight.shape)
                weights.append(weight)

        y = np.array(y, dtype=np.double)
        w = np.array(weights, dtype=np.double)
        obs = self.data[idx]['obs']
        obs['agents'] = obs['agents'][self.skip_agents:self.stop_agents]
        return {'obs': self.data[idx]['obs']}, {'y': y, 'w': w}

class PathplanningDataset(BaseDataset):
    def __init__(self, path, config):
        # cov_only: Predict only coverage or predict both coverage and map
        # is_global: Predict local coverage and map or global coverage and map

        super().__init__(path)
        self.is_relative=config['format']=='relative'
        self.pred_mode=config['prediction']
        self.is_global=config['pred_mode']=='global'
        self.skip_agents=config['skip_agents'] if 'skip_agents' in config else 0
        self.stop_agents=config['stop_agents'] if 'stop_agents' in config else None

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        y = []
        weights = []
        for agent_obs in self.data[idx]['obs']['agents'][self.skip_agents:self.stop_agents]:
            if self.is_global:
                obs_map = np.zeros(self.data[idx]['obs']['state'].shape[:2], dtype=np.float)
                obs_pos = np.zeros(self.data[idx]['obs']['state'].shape[:2], dtype=np.float)
                obs_goal = np.zeros(self.data[idx]['obs']['state'].shape[:2], dtype=np.float)
                obs_map[self.data[idx]['obs']['state'][..., 0] == 1] = 1
                for i in range(self.data[idx]['obs']['state'].shape[-1]):
                    obs_pos[self.data[idx]['obs']['state'][..., i] == 2] = 1
                    obs_goal[self.data[idx]['obs']['state'][..., i] == 3] = 1

                if self.is_relative:
                    obs_goal = self.to_agent_coord_frame(obs_goal, self.world_shape[0]*2, agent_obs['pos'], fill=0)
                    obs_pos = self.to_agent_coord_frame(obs_pos, self.world_shape[0]*2, agent_obs['pos'], fill=0)
                    obs_map = self.to_agent_coord_frame(obs_map, self.world_shape[0]*2, agent_obs['pos'], fill=1)
            else:
                # directly use agent's relative view coverage
                obs_map = agent_obs['map'][..., 0]
                obs_goal = agent_obs['map'][..., 1]
                obs_pos = agent_obs['map'][..., 2]

            if self.pred_mode == "goal":
                # only predict local coverage and use world map as mask
                y.append(np.stack([obs_goal], axis=0))
                weight = (~obs_map.astype(np.bool)).astype(np.int)
                # goal can generally be on the margin if it is projected!
                for row in [0, -1]:
                    weight[row] = 1
                    weight[:, row] = 1
                weights.append(copy.deepcopy([weight]))
            elif self.pred_mode == "all":
                # predict both local coverage and world map, but mask out everything outside the world shifted to the agents position
                d = np.stack([obs_map, obs_goal, obs_pos], axis=0)
                y.append(d)
                weight = np.ones(obs_map.shape)
                weight = np.stack([weight]*3, axis=0)
                weights.append(weight)

        y = np.array(y, dtype=np.double)
        w = np.array(weights, dtype=np.double)
        obs = self.data[idx]['obs']
        obs['agents'] = obs['agents'][self.skip_agents:self.stop_agents]
        return {'obs': self.data[idx]['obs']}, {'y': y, 'w': w}

dataset_classes = {
    "path": PathplanningDataset,
    "cov": CoverageDataset
}

def inference(model_checkpoint_path,
              data_path,
              seed=None, run_eval=False, save_dirname=None):
    if seed is None:
        seed = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    batch_size = 1

    checkpoint_file = Path(model_checkpoint_path)
    with open(checkpoint_file.parent / 'config.json', 'r') as config_file:
        config = json.load(config_file)
    config['skip_agents'] = 0
    dataset = dataset_classes[config['type']](data_path, config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    model = load_model(checkpoint_file, Model(dataset, config))

    cmap_map = colors.LinearSegmentedColormap.from_list("cmap_map", [(0, 0, 0, 0), (0, 0, 0, 1)])
    cmap_cov = colors.LinearSegmentedColormap.from_list("cmap_cov", [(0, 0, 0, 0), (0, 1, 0, 1)])
    cmap_own_cov = colors.LinearSegmentedColormap.from_list("cmap_cov", [(0, 0, 0, 0), (1, 1, 0, 1)])

    def transform_rel_abs(data, pos):
        return np.roll(data, pos, axis=(0, 1))[int(dataset.obs_size / 2):, int(dataset.obs_size / 2):]

    if run_eval:
        evaluator = create_evaluator(model)
        evaluator.run(loader)
        metrics = evaluator.state.metrics
        print(metrics)
    identifier = f"{config['format']}-{config['nn_mode']}-{config['pred_mode']}"
    batch_index = 0
    for x, y_true in loader:
        fig, axs = plt.subplots(batch_size*2, 5, figsize=[8, 3.2])
        if run_eval:
            #axs[0][0].set_title(f"ap: {metrics['ap']:.4f}, auc: {metrics['auc']:.4f}")
            fig.suptitle(f"{identifier} f1: {metrics['f1']:.4f}, ap: {metrics['ap']:.4f}, auc: {metrics['auc']:.4f}")

        y_pred = model(x).detach().numpy()
        #y_pred = torch.round(model(x).detach()).numpy()
        for i in range(batch_size):
            for j in range(5):
                #for agent in x['obs']['agents']:
                #    print(agent['pos'])

                #axs[i*2][j].imshow(y_true['y'][i][j][1, :, :], cmap=cmap_cov)
                agent_obs = x['obs']['agents'][j]
                pos = agent_obs['pos'][i]
                agent_map = agent_obs['map'][i]

                if config['format'] == 'relative':
                    #axs[i*2][j].imshow(transform_rel_abs(agent_map[...,0], pos), cmap=cmap_map) # obstacles
                    #axs[i*2][j].imshow(transform_rel_abs(y_true['y'][i][j][0, :, :], pos), cmap=cmap_cov)
                    #axs[i*2][j].imshow(transform_rel_abs(agent_map[...,1], pos), cmap=cmap_own_cov)

                    axs[i*2][j].imshow(agent_map[...,0], cmap=cmap_map) # obstacles
                    #axs[i*2][j].imshow(y_true['y'][i][j][1, :, :], cmap=cmap_map)
                    axs[i*2][j].imshow(y_true['y'][i][j][0, :, :], cmap=cmap_cov)

                    #print(y_pred[i][j][1, :, :])
                    #axs[i*2+1][j].imshow(y_pred[i][j][1, :, :], cmap=cmap_map)
                    #axs[i*2+1][j].imshow(y_pred[i][j][0, :, :], cmap=cmap_cov)

                    #axs[i*2][j].imshow(agent_map[...,1], cmap=cmap_own_cov)

                    axs[i*2+1][j].imshow(y_pred[i][j][0, :, :], cmap=cmap_cov)
                    #axs[i*2+1][j].imshow(y_pred[i][j][1, :, :], cmap=cmap_map)
                    axs[i * 2+1][j].imshow(agent_map[..., 0], cmap=cmap_map)  # obstacles
                    #axs[i*2+1][j].imshow(y_true['w'][i][j][0, :, :], cmap=cmap_map) # weighting
                    #axs[i*2+1][j].imshow(transform_rel_abs(y_pred[i][j][0, :, :], pos), cmap=cmap_own_cov if config['pred_mode'] == 'local' else cmap_cov)

                    #map_data = transform_rel_abs(agent_obs['map'][i][...,0], pos) if len(y_pred[i][j]) == 1 else y_pred[i][j][1, :, :]
                    #axs[i*2+1][j].imshow(map_data, cmap=cmap_map) # obstacles

                else:
                    axs[i*2][j].imshow(x['obs']['state'][i][...,0], cmap=cmap_map) # obstacles
                    axs[i*2][j].imshow(x['obs']['state'][i][...,1], cmap=cmap_cov)
                    m = np.roll(agent_obs['map'][i], agent_obs['pos'][i], axis=(0,1))[int(dataset.obs_size/2):,int(dataset.obs_size/2):]
                    axs[i*2][j].imshow(m[...,1], cmap=cmap_own_cov)

                    axs[i*2+1][j].imshow(y_pred[i][j][0, :, :], cmap=cmap_own_cov if config['pred_mode'] == 'local' else cmap_cov)
                    axs[i*2+1][j].imshow(x['obs']['state'][i][...,0], cmap=cmap_map) # obstacles

                '''
                for k in range(2):
                    rect = patches.Rectangle((agent_obs['pos'][i][1] - 1 / 2, agent_obs['pos'][i][0] - 1 / 2), 1, 1,
                                             linewidth=1, edgecolor='r', facecolor='none')
                    axs[i*2+k][j].add_patch(rect)
                '''
                for k in range(2):
                    axs[i * 2 + k][j].set_xticks([])
                    axs[i * 2 + k][j].set_yticks([])

        fig.tight_layout() #rect=[0, 0.03, 1, 0.95])
        if save_dirname is not None:
            img_path = checkpoint_file.parent/save_dirname
            img_path.mkdir(exist_ok=True)
            frame_path = img_path/f"{batch_index:05d}.png"
            print("Frame", frame_path)
            plt.savefig(frame_path, dpi=300)
        else:
            plt.show()
        plt.close()
        batch_index += 1
        if batch_index == 300:
            break

def inference_gnn_cnn(cnn_model_checkpoint_path,
                      gnn_model_checkpoint_path,
                      data_path,
                      seed=None, run_eval=False, save_dirname=None):
    if seed is None:
        seed = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    batch_size = 1

    checkpoint_file = Path(gnn_model_checkpoint_path)
    with open(checkpoint_file.parent / 'config.json', 'r') as config_file:
        config = json.load(config_file)
    config['skip_agents'] = 0
    gnn_dataset = dataset_classes[config['type']](data_path, config)
    gnn_loader = torch.utils.data.DataLoader(
        gnn_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )
    gnn_model = load_model(checkpoint_file, Model(gnn_dataset, config))

    checkpoint_file = Path(cnn_model_checkpoint_path)
    with open(checkpoint_file.parent / 'config.json', 'r') as config_file:
        config = json.load(config_file)
    config['skip_agents'] = 0
    cnn_dataset = dataset_classes[config['type']](data_path, config)
    cnn_loader = torch.utils.data.DataLoader(
        cnn_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )
    cnn_model = load_model(checkpoint_file, Model(cnn_dataset, config))


    cmap_map = colors.LinearSegmentedColormap.from_list("cmap_map", [(0, 0, 0, 0), (0, 0, 0, 1)])
    cmap_cov = colors.LinearSegmentedColormap.from_list("cmap_cov", [(0, 0, 0, 0), (0, 1, 0, 1)])
    cmap_own_cov = colors.LinearSegmentedColormap.from_list("cmap_cov", [(0, 0, 0, 0), (0, 0, 1, 1)])

    def transform_rel_abs(data, pos):
        return np.roll(data, pos, axis=(0, 1))[int(gnn_dataset.obs_size / 2):, int(gnn_dataset.obs_size / 2):]

    identifier = f"{config['format']}-{config['nn_mode']}-{config['pred_mode']}"
    batch_index = 0
    for (x, y_true), (x_cnn, y_cnn_true) in zip(gnn_loader, cnn_loader):
        fig, axs = plt.subplots(batch_size*2, 5, figsize=[8, 3.2])

        y_pred = gnn_model(x).detach().numpy()
        y_pred_cnn = cnn_model(x_cnn).detach().numpy()
        for i in range(batch_size):
            for j in range(5):
                if j == 0:
                    agent_obs = x['obs']['agents'][j]
                    pos = agent_obs['pos'][i]
                    agent_map = agent_obs['map'][i]
                else:
                    agent_obs = x_cnn['obs']['agents'][j]
                    pos = agent_obs['pos'][i]
                    agent_map = agent_obs['map'][i]

                axs[i*2][j].imshow(transform_rel_abs(agent_map[...,0], pos), cmap=cmap_map) # obstacles
                axs[i*2][j].imshow(transform_rel_abs(y_true['y'][i][j][0, :, :], pos), cmap=cmap_cov)
                axs[i*2][j].imshow(transform_rel_abs(agent_map[...,1], pos), cmap=cmap_own_cov)

                if j == 0:
                    axs[i*2+1][j].imshow(transform_rel_abs(y_pred_cnn[i][j][0, :, :], pos), cmap=cmap_own_cov)
                else:
                    axs[i*2+1][j].imshow(transform_rel_abs(y_pred[i][j][0, :, :], pos), cmap=cmap_cov)

                map_data = agent_obs['map'][i][...,0] if len(y_pred[i][j]) == 1 else y_pred[i][j][1, :, :]
                axs[i*2+1][j].imshow(transform_rel_abs(map_data, pos), cmap=cmap_map) # obstacles

                for k in range(2):
                    rect = patches.Rectangle((agent_obs['pos'][i][1] - 1 / 2, agent_obs['pos'][i][0] - 1 / 2), 1, 1,
                                             linewidth=1, edgecolor='r', facecolor='none')
                    axs[i*2+k][j].add_patch(rect)

                for k in range(2):
                    axs[i * 2 + k][j].set_xticks([])
                    axs[i * 2 + k][j].set_yticks([])

        fig.tight_layout() #rect=[0, 0.03, 1, 0.95])
        if save_dirname is not None:
            img_path = checkpoint_file.parent/save_dirname
            img_path.mkdir(exist_ok=True)
            frame_path = img_path/f"{batch_index:05d}.png"
            print("Frame", frame_path)
            plt.savefig(frame_path, dpi=300)
        else:
            plt.show()
        plt.close()
        batch_index += 1
        if batch_index == 10:
            break

def inference_path(model_checkpoint_path,
              data_path,
              seed=None, run_eval=False, save_dirname=None):
    if seed is None:
        seed = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    batch_size = 1

    checkpoint_file = Path(model_checkpoint_path)
    with open(checkpoint_file.parent / 'config.json', 'r') as config_file:
        config = json.load(config_file)
    config['skip_agents'] = 0
    dataset = dataset_classes[config['type']](data_path, config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    model = load_model(checkpoint_file, Model(dataset, config))

    cmap_map = colors.LinearSegmentedColormap.from_list("cmap_map", [(0, 0, 0, 0), (0, 0, 0, 1)])
    cmap_pos = colors.LinearSegmentedColormap.from_list("cmap_cov", [(0, 0, 0, 0), (0, 1, 0, 1)])
    cmap_goal = colors.LinearSegmentedColormap.from_list("cmap_cov", [(0, 0, 0, 0), (1, 1, 0, 1)])

    def transform_rel_abs(data, pos):
        return data #np.roll(data, pos, axis=(0, 1))#[int(dataset.obs_size / 2):, int(dataset.obs_size / 2):]

    if run_eval:
        evaluator = create_evaluator(model)
        evaluator.run(loader)
        metrics = evaluator.state.metrics
        print(metrics)
    identifier = f"{config['format']}-{config['nn_mode']}-{config['pred_mode']}"
    batch_index = 0
    for x, y_true in loader:
        n_agents = len(x['obs']['agents'])
        fig, axs = plt.subplots(batch_size*2, 5, figsize=[8, 3.2])
        if run_eval:
            #axs[0][0].set_title(f"ap: {metrics['ap']:.4f}, auc: {metrics['auc']:.4f}")
            fig.suptitle(f"{identifier} f1: {metrics['f1']:.4f}, ap: {metrics['ap']:.4f}, auc: {metrics['auc']:.4f}")

        y_pred = model(x).detach().numpy()
        #y_pred = torch.round(model(x).detach()).numpy()
        for i in range(batch_size):
            for j in range(5):
                #for agent in x['obs']['agents']:
                #    print(agent['pos'])

                #axs[i*2][j].imshow(y_true['y'][i][j][1, :, :], cmap=cmap_cov)
                agent_obs = x['obs']['agents'][j]
                pos = agent_obs['pos'][i]
                agent_map = agent_obs['map'][i]

                axs[i*2][j].imshow(transform_rel_abs(agent_map[...,0], pos), cmap=cmap_map) # obstacles
                axs[i*2][j].imshow(transform_rel_abs(agent_map[...,2], pos), cmap=cmap_pos) # obstacles
                axs[i*2][j].imshow(transform_rel_abs(agent_map[...,1], pos), cmap=cmap_goal) # obstacles

                axs[i*2+1][j].imshow(transform_rel_abs(agent_map[...,0], pos), cmap=cmap_map) # obstacles
                axs[i*2+1][j].imshow(transform_rel_abs(agent_map[...,2], pos), cmap=cmap_pos) # obstacles
                axs[i*2+1][j].imshow(y_pred[i][j][0, :, :], cmap=cmap_goal)

                #axs[i*2+1][j].imshow(y_pred[i][j][1, :, :], cmap=cmap_goal)
                #axs[i*2+1][j].imshow(y_pred[i][j][2, :, :], cmap=cmap_pos)

                #axs[i*2+1][j].imshow(transform_rel_abs(y_pred[i][j][0, :, :], pos), cmap=cmap_own_cov if config['pred_mode'] == 'local' else cmap_cov)
                #axs[i*2+1][j].imshow(transform_rel_abs(agent_obs['map'][i][...,0], pos), cmap=cmap_map) # obstacles

                for k in range(2):
                    axs[i * 2 + k][j].set_xticks([])
                    axs[i * 2 + k][j].set_yticks([])

        fig.tight_layout() #rect=[0, 0.03, 1, 0.95])
        if save_dirname is not None:
            img_path = checkpoint_file.parent/save_dirname
            img_path.mkdir(exist_ok=True)
            frame_path = img_path/f"{batch_index:05d}.png"
            print("Frame", frame_path)
            plt.savefig(frame_path, dpi=300)
        else:
            plt.show()
        plt.close()
        batch_index += 1
        if batch_index == 300:
            break

def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y

def apply_weight_output_transform(x):
    y_pred_raw, y_raw = x[0], x[1] # shape each (batch size, agent, channel, x, y)

    classes = y_pred_raw.shape[2]

    w = y_raw['w'].permute([2, 0,1,3,4]).flatten()
    y = y_raw['y'].permute([2, 0,1,3,4]).flatten()[w==1].reshape(-1, classes)
    y_pred = y_pred_raw.permute([2, 0,1,3,4]).flatten()[w==1].reshape(-1, classes)

    return (y_pred, y)

def apply_weight_threshold_output_transform(x):
    return apply_weight_output_transform(thresholded_output_transform(x))

def weighted_binary_cross_entropy(y_pred, y):
    return F.binary_cross_entropy(y_pred, y['y'], weight=y['w'])

from ignite.metrics import EpochMetric

class AveragePrecision(EpochMetric):
    def __init__(self, output_transform=lambda x: x):
        def average_precision_compute_fn(y_preds, y_targets):
            try:
                from sklearn.metrics import average_precision_score
            except ImportError:
                raise RuntimeError("This contrib module requires sklearn to be installed.")

            y_true = y_targets.numpy()
            y_pred = y_preds.numpy()
            return average_precision_score(y_true, y_pred, average='micro')

        super(AveragePrecision, self).__init__(average_precision_compute_fn, output_transform=output_transform)

def create_evaluator(model):
    return create_supervised_evaluator(
        model,
        metrics={
            #"p": Precision(apply_weight_threshold_output_transform),
            #"r": Recall(apply_weight_threshold_output_transform),
            #"f1": Fbeta(1, output_transform=apply_weight_threshold_output_transform),
            #"auc": ROC_AUC(output_transform=apply_weight_output_transform),
            "ap": AveragePrecision(output_transform=apply_weight_output_transform)
        },
        device=model.device
    )

def load_model(checkpoint_path, model):
    model_state = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # A basic remapping is required
    mapping = {k: v for k, v in zip(model_state.keys(), model.state_dict().keys())}
    mapped_model_state = OrderedDict([(mapping[k], v) for k, v in model_state.items()])
    model.load_state_dict(mapped_model_state, strict=False)
    return model

def train(train_data_path,
          valid_data_path,
          config,
          out_dir="./explainability",
          batch_size=64,
          lr=1e-4,
          epochs=100):
    train_dataset = dataset_classes[config['type']](train_data_path, config)
    valid_dataset = dataset_classes[config['type']](valid_data_path, config)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    model = Model(valid_dataset, config)
    path_cp = "./explainability_checkpoints/"+out_dir
    os.makedirs(path_cp, exist_ok=True)
    with open(path_cp+"/config.json", 'w') as config_file:
        json.dump(config, config_file)

    optimizer = Adam(model.parameters(), lr=lr)
    trainer = create_supervised_trainer(model, optimizer, weighted_binary_cross_entropy, device=model.device)

    validation_evaluator = create_evaluator(model)

    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    best_model_handler = ModelCheckpoint(dirname="./explainability_checkpoints/"+out_dir,
                                         filename_prefix="best",
                                         n_saved=1,
                                         global_step_transform=global_step_from_engine(trainer),
                                         score_name="val_ap",
                                         score_function=lambda engine: engine.state.metrics['ap'],
                                         require_empty=False)
    validation_evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model, })

    tb_logger = TensorboardLogger(log_dir='./explainability_tensorboard/'+out_dir)
    tb_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="training", output_transform=lambda loss: {"batchloss": loss}, metric_names="all"
        ),
        event_name=Events.ITERATION_COMPLETED(every=100),
    )

    tb_logger.attach(
        validation_evaluator,
        log_handler=OutputHandler(tag="validation", metric_names=["ap"], another_engine=trainer),
        event_name=Events.EPOCH_COMPLETED,
    )
    #tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_COMPLETED(every=100))
    #tb_logger.attach(trainer, log_handler=WeightsScalarHandler(model), event_name=Events.ITERATION_COMPLETED(every=100))
    #tb_logger.attach(trainer, log_handler=WeightsHistHandler(model), event_name=Events.EPOCH_COMPLETED(every=100))
    #tb_logger.attach(trainer, log_handler=GradsScalarHandler(model), event_name=Events.ITERATION_COMPLETED(every=100))
    #tb_logger.attach(trainer, log_handler=GradsHistHandler(model), event_name=Events.EPOCH_COMPLETED(every=100))

    @trainer.on(Events.EPOCH_COMPLETED(every=5))
    def log_validation_results(engine):
        validation_evaluator.run(val_loader)
        metrics = validation_evaluator.state.metrics
        pbar.log_message(
            f"Validation Results - Epoch: {engine.state.epoch} ap: {metrics['ap']}" # f1: {metrics['f1']}, p: {metrics['p']}, r: {metrics['r']}
        )

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)

def evaluate(model_checkpoint, data_path, **kwargs):
    checkpoint_file = Path(model_checkpoint)
    with open(checkpoint_file.parent/'config.json', 'r') as config_file:
        config = json.load(config_file)

    ap = []
    for start, end in [[0, 1], [1, None]]:
        config['stop_agents'] = end
        config['skip_agents'] = start

        dataset = dataset_classes[config['type']](data_path, config)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=2
        )
        model = load_model(checkpoint_file, Model(dataset, config))
        evaluator = create_evaluator(model)
        t0 = time.time()
        evaluator.run(loader)
        #print("T", time.time() - t0)
        m = evaluator.state.metrics['ap']
        if not isinstance(m, list):
            m = [m]
        ap.append(m)
    print(ap)
    return ap

def analyse_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    analyse(data)

    def get_coverage_fraction(map, coverage):
        coverable_area = ~(map > 0)
        covered_area = coverage & coverable_area
        return np.sum(covered_area) / np.sum(coverable_area)

    coverage_fractions = []
    for sample in data:
        world_cov = sample['obs']['state'][..., 1]
        world_map = sample['obs']['state'][..., 0]
        coverage_fractions.append(get_coverage_fraction(world_map, world_cov))

    print(np.mean(coverage_fractions), np.std(coverage_fractions))
#    plt.hist(coverage_fractions)
#    plt.show()


if __name__ == "__main__":
    #train("explainability_data_k3_268ugnliyw_2735_train.pkl", "explainability_data_k3_268ugnliyw_2735_valid.pkl", "./explainability_k3_sgd", epochs=10000, batch_size=32, lr=0.1, sgd_momentum=0.9)
    if False:
        #dataset_id = "031g2r0u73_3070"
        #dataset_id = "096dwc6v_g_2990"
        #dataset_id = "22hfzad070_3050"
        #dataset_id = "280kl0nkhl_2960"
        #dataset_id = "44cdqovnq4_1540" # split
        #dataset_id = "27c8iboc7__2250" # flow

        #dataset_id = "101jtn1ssr_2450"
        #dataset_id = "46t6qhvrxf_5400"

        dataset_id = "300d3g1xqj_3120"
        #dataset_id = "303qg71k5o_3120"
        train(
            f"/local/scratch/jb2270/datasets_corl/explainability_data_{dataset_id}_train.pkl",
            f"/local/scratch/jb2270/datasets_corl/explainability_data_{dataset_id}_valid.pkl",
            {
                'format': 'relative', # absolute/relative
                'nn_mode': 'cnn', # cnn/gnn/gnn_cnn
                'pred_mode': 'local', # local (own coverage and map)/global (global coverage and map)
                'prediction': 'cov_map', #cov_only/cov_map
                'type': 'cov' # path/cov
            },
            f"explainability_cov_map_local_cnn_{dataset_id}",
            epochs=10000,
            batch_size=64,
            lr=5e-3,
        )

    #evaluate("explainability_data_56uhj2ync9_2650_valid.pkl", "./explainability_checkpoints/explainability_56uhj2ync9_rel_glob/best_model_636_val_auc=0.8963244891793261.pth")

    #inference("./results/0610/explainability_checkpoints/explainability_228pbizcxq_1955_glob/best_model_973_val_auc=0.8896503235407915.pth", "./explainability_data_228pbizcxq_1955_test.pkl", 11, False)

    #inference("./results/0610/explainability_checkpoints/explainability_228pbizcxq_1955_loc/best_model_1017_val_auc=0.9856628585466523.pth", "./explainability_data_228pbizcxq_1955_test.pkl", 11, False)

    # flow
    #evaluate("./results/0712/explainability_checkpoints/explainability_glob_27c8iboc7__2250/best_model_162_val_auc=0.9998915352938512.pth", "./results/0712/explainability_data_27c8iboc7__3500_comm_test.pkl", save_dirname="rendering_comm")
    #evaluate("./results/0712/explainability_checkpoints/explainability_glob_27c8iboc7__2250/best_model_162_val_auc=0.9998915352938512.pth", "./results/0712/explainability_data_27c8iboc7__3500_nocomm_test.pkl", save_dirname="rendering_nocomm")
    #inference_path("./results/0721/explainability_checkpoints/explainability_path_goal_only_local_cnn_27c8iboc7__2250/best_model_30_val_ap=1.0.pth", "./results/0712/explainability_data_27c8iboc7__3500_nocomm_test.pkl")

    # split
    #inference("./results/0712/explainability_checkpoints/explainability_cov_map_44cdqovnq4_1540/best_model_77_val_auc=0.999701756785555.pth", "./results/0712/explainability_data_44cdqovnq4_1540_nocomm_test.pkl")
    #evaluate("./results/0712/explainability_checkpoints/explainability_cov_map_44cdqovnq4_1540/best_model_77_val_auc=0.999701756785555.pth", "./results/0712/explainability_data_44cdqovnq4_1540_comm_test.pkl", save_dirname="rendering_comm")
    #evaluate("./results/0712/explainability_checkpoints/explainability_cov_map_global_gnn_44cdqovnq4_1540/best_model_67_val_auc=0.9980259594481737.pth", "./results/0712/explainability_data_44cdqovnq4_1540_comm_test.pkl", save_dirname="rendering_comm")
    #evaluate("./results/0712/explainability_checkpoints/explainability_cov_map_global_gnn_44cdqovnq4_1540/best_model_67_val_auc=0.9980259594481737.pth", "./results/0712/explainability_data_44cdqovnq4_1540_nocomm_test.pkl", save_dirname="rendering_nocomm")

    # coverage normal
    #evaluate("./results/0712/explainability_checkpoints/explainability_cov_map_local_cnn_300d3g1xqj_3120/best_model_139_val_auc=0.9885640826508932.pth", "./results/0712/explainability_data_300d3g1xqj_3120_nocomm_test.pkl", save_dirname="rendering_nocomm")
    #inference("./results/0712/explainability_checkpoints/explainability_cov_map_local_cnn_300d3g1xqj_3120/best_model_139_val_auc=0.9885640826508932.pth", "./results/0712/explainability_data_300d3g1xqj_3120_comm_test.pkl") #, save_dirname="rendering_comm")
    #inference("./results/0721/explainability_checkpoints/explainability_cov_cov_only_local_cnn_300d3g1xqj_3120/best_model_190_val_ap=0.8773743058356964.pth", "./results/0712/explainability_data_300d3g1xqj_3120_comm_test.pkl") #, save_dirname="rendering_comm")
    #evaluate("./results/0721/explainability_checkpoints/explainability_cov_cov_only_local_cnn_300d3g1xqj_3120/best_model_190_val_ap=0.8773743058356964.pth", "./results/0712/explainability_data_300d3g1xqj_3120_comm_test.pkl") #, save_dirname="rendering_comm")

    #inference("./results/0721/explainability_checkpoints/explainability_cov_cov_only_global_gnn_300d3g1xqj_3120/best_model_140_val_ap=0.8570488698746233.pth", "./results/0712/explainability_data_300d3g1xqj_3120_comm_test.pkl") #, save_dirname="rendering_comm")
    #inference_gnn_cnn(
    #    "./results/0721/explainability_checkpoints/explainability_cov_cov_only_local_cnn_300d3g1xqj_3120/best_model_190_val_ap=0.8773743058356964.pth",
    #    "./results/0721/explainability_checkpoints/explainability_cov_cov_only_global_gnn_300d3g1xqj_3120/best_model_140_val_ap=0.8570488698746233.pth",
    #    "./results/0712/explainability_data_300d3g1xqj_3120_comm_test.pkl",
    #    save_dirname="rendering_comm"
    #)
    #evaluate("./results/0712/explainability_checkpoints/explainability_cov_map_global_gnn_300d3g1xqj_3120/best_model_69_val_auc=0.961077006252264.pth", "./results/0712/explainability_data_300d3g1xqj_3120_nocomm_test.pkl", save_dirname="rendering_nocomm")
    #inference("./results/0712/explainability_checkpoints/explainability_cov_map_global_gnn_300d3g1xqj_3120/best_model_69_val_auc=0.961077006252264.pth", "./results/0712/explainability_data_300d3g1xqj_3120_comm_test.pkl") #,save_dirname="rendering_comm")
    #inference("./results/0712/explainability_checkpoints/explainability_cov_map_global_gnn_300d3g1xqj_3120/best_model_69_val_auc=0.961077006252264.pth", "./results/0712/explainability_data_300d3g1xqj_3120_comm_test.pkl") #,save_dirname="rendering_comm")

    #evaluate("./results/0712/explainability_checkpoints/explainability_300d3g1xqj_3120/best_model_384_val_auc=0.9779040424681612.pth", "./results/0712/explainability_data_300d3g1xqj_3120_nocomm_test.pkl", save_dirname="rendering_nocomm")
    #evaluate("./results/0712/explainability_checkpoints/explainability_300d3g1xqj_3120/best_model_384_val_auc=0.9779040424681612.pth", "./results/0712/explainability_data_300d3g1xqj_3120_comm_test.pkl", save_dirname="rendering_comm")

    #inference("./results/0823/expl_checkpoints/explainability_cov_cov_map_local_271e7f5bc3_1560/best_model_30_val_ap=0.9611180740842954.pth", "../../Internship/gpvae/data/explainability_data_271e7f5bc3_1560_test.pkl") #,save_dirname="rendering_comm")
    inference("./results/0823/expl_checkpoints/explainability_cov_cov_only_local_271e7f5bc3_1560/best_model_85_val_ap=0.8428215464016102.pth", "../../Internship/gpvae/data/explainability_data_271e7f5bc3_1560_test.pkl") #,save_dirname="rendering_comm")
