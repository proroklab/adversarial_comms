from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.models.torch.misc import normc_initializer, same_padding, SlimConv2d, SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

from .gnn import adversarialGraphML as gml_adv
from .gnn import graphML as gml
from .gnn import graphTools
import numpy as np
import copy

torch, nn = try_import_torch()
from torchsummary import summary

# https://ray.readthedocs.io/en/latest/using-ray-with-pytorch.html

DEFAULT_OPTIONS = {
    "activation": "relu",
    "agent_split": 1,
    "cnn_compression": 512,
    "cnn_filters": [[32, [8, 8], 4], [64, [4, 4], 2], [128, [4, 4], 2]],
    "cnn_residual": False,
    "freeze_coop": True,
    "freeze_coop_value": False,
    "freeze_greedy": False,
    "freeze_greedy_value": False,
    "graph_edge_features": 1,
    "graph_features": 512,
    "graph_layers": 1,
    "graph_tabs": 3,
    "relative": True,
    "value_cnn_compression": 512,
    "value_cnn_filters": [[32, [8, 8], 2], [64, [4, 4], 2], [128, [4, 4], 2]],
    "forward_values": True
}

class AdversarialModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):#,
        #graph_layers, graph_features, graph_tabs, graph_edge_features, cnn_filters, value_cnn_filters, value_cnn_compression, cnn_compression, relative, activation):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(model_config['custom_model_config'])

        #self.cfg = model_config['custom_options']
        self.n_agents = len(obs_space.original_space['agents'])
        self.graph_features = self.cfg['graph_features']
        self.cnn_compression = self.cfg['cnn_compression']
        self.activation = {
            'relu': nn.ReLU,
            'leakyrelu': nn.LeakyReLU
        }[self.cfg['activation']]

        layers = []
        input_shape = obs_space.original_space['agents'][0]['map'].shape
        (w, h, in_channels) = input_shape

        in_size = [w, h]
        for out_channels, kernel, stride in self.cfg['cnn_filters'][:-1]:
            padding, out_size = same_padding(in_size, kernel, [stride, stride])
            layers.append(SlimConv2d(in_channels, out_channels, kernel, stride, padding, activation_fn=self.activation))
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = self.cfg['cnn_filters'][-1]
        layers.append(
            SlimConv2d(in_channels, out_channels, kernel, stride, None))
        layers.append(nn.Flatten(1, -1))
        #if isinstance(cnn_compression, int):
        #    layers.append(nn.Linear(cnn_compression, self.cfg['graph_features']-2)) # reserve 2 for pos
        #    layers.append(self.activation{))
        self.coop_convs = nn.Sequential(*layers)
        self.greedy_convs = copy.deepcopy(self.coop_convs)

        self.coop_value_obs_convs = copy.deepcopy(self.coop_convs)
        self.greedy_value_obs_convs = copy.deepcopy(self.coop_convs)

        summary(self.coop_convs, device="cpu", input_size=(input_shape[2], input_shape[0], input_shape[1]))

        gfl = []
        for i in range(self.cfg['graph_layers']):
            gfl.append(gml_adv.GraphFilterBatchGSOA(self.graph_features, self.graph_features, self.cfg['graph_tabs'], self.cfg['agent_split'], self.cfg['graph_edge_features'], False))
            #gfl.append(gml.GraphFilterBatchGSO(self.graph_features, self.graph_features, self.cfg['graph_tabs'], self.cfg['graph_edge_features'], False))
            gfl.append(self.activation())

        self.GFL = nn.Sequential(*gfl)

        #gso_sum = torch.zeros(2, 1, 8, 8)
        #self.GFL[0].addGSO(gso_sum)
        #summary(self.GFL, device="cuda" if torch.cuda.is_available() else "cpu", input_size=(self.graph_features, 8))

        logits_inp_features = self.graph_features
        if self.cfg['cnn_residual']:
            logits_inp_features += self.cnn_compression

        post_logits = [
            nn.Linear(logits_inp_features, 64),
            self.activation(),
            nn.Linear(64, 32),
            self.activation()
        ]
        logit_linear = nn.Linear(32, 5)
        nn.init.xavier_uniform_(logit_linear.weight)
        nn.init.constant_(logit_linear.bias, 0)
        post_logits.append(logit_linear)
        self.coop_logits = nn.Sequential(*post_logits)
        self.greedy_logits = copy.deepcopy(self.coop_logits)
        summary(self.coop_logits, device="cpu", input_size=(logits_inp_features,))

        ##############################

        layers = []
        input_shape = np.array(obs_space.original_space['state'].shape)
        (w, h, in_channels) = input_shape

        in_size = [w, h]
        for out_channels, kernel, stride in self.cfg['value_cnn_filters'][:-1]:
            padding, out_size = same_padding(in_size, kernel, [stride, stride])
            layers.append(SlimConv2d(in_channels, out_channels, kernel, stride, padding, activation_fn=self.activation))
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = self.cfg['value_cnn_filters'][-1]
        layers.append(
            SlimConv2d(in_channels, out_channels, kernel, stride, None))
        layers.append(nn.Flatten(1, -1))

        self.coop_value_cnn = nn.Sequential(*layers)
        self.greedy_value_cnn = copy.deepcopy(self.coop_value_cnn)
        summary(self.greedy_value_cnn, device="cpu", input_size=(input_shape[2], input_shape[0], input_shape[1]))

        layers = [
            nn.Linear(self.cnn_compression + self.cfg['value_cnn_compression'], 64),
            self.activation(),
            nn.Linear(64, 32),
            self.activation()
        ]
        values_linear = nn.Linear(32, 1)
        normc_initializer()(values_linear.weight)
        nn.init.constant_(values_linear.bias, 0)
        layers.append(values_linear)

        self.coop_value_branch = nn.Sequential(*layers)
        self.greedy_value_branch = copy.deepcopy(self.coop_value_branch)
        summary(self.coop_value_branch, device="cpu", input_size=(self.cnn_compression + self.cfg['value_cnn_compression'],))

        self._cur_value = None

        self.freeze_coop_value(self.cfg['freeze_coop_value'])
        self.freeze_greedy_value(self.cfg['freeze_greedy_value'])
        self.freeze_coop(self.cfg['freeze_coop'])
        self.freeze_greedy(self.cfg['freeze_greedy'])

    def freeze_coop(self, freeze):
        all_params = \
            list(self.coop_convs.parameters()) + \
            [self.GFL[0].weight1] + \
            list(self.coop_logits.parameters())

        for param in all_params:
            param.requires_grad = not freeze

    def freeze_greedy(self, freeze):
        all_params = \
            list(self.greedy_logits.parameters()) + \
            list(self.greedy_convs.parameters()) + \
            [self.GFL[0].weight0]

        for param in all_params:
            param.requires_grad = not freeze

    def freeze_greedy_value(self, freeze):
        all_params = \
            list(self.greedy_value_branch.parameters()) + \
            list(self.greedy_value_cnn.parameters()) + \
            list(self.greedy_value_obs_convs)

        for param in all_params:
            param.requires_grad = not freeze

    def freeze_coop_value(self, freeze):
        all_params = \
            list(self.coop_value_cnn.parameters()) + \
            list(self.coop_value_branch.parameters()) + \
            list(self.coop_value_obs_convs)

        for param in all_params:
            param.requires_grad = not freeze

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        batch_size = input_dict["obs"]['gso'].shape[0]
        o_as = input_dict["obs"]['agents']

        gso = input_dict["obs"]['gso'].unsqueeze(1)
        device = gso.device

        for i in range(len(self.GFL)//2):
            self.GFL[i*2].addGSO(gso)

        greedy_cnn = self.greedy_convs(o_as[0]['map'].permute(0, 3, 1, 2))
        coop_agents_cnn = {id_agent: self.coop_convs(o_as[id_agent]['map'].permute(0, 3, 1, 2)) for id_agent in range(1, len(o_as))}

        greedy_value_obs_cnn = self.greedy_value_obs_convs(o_as[0]['map'].permute(0, 3, 1, 2))
        coop_value_obs_cnn = {id_agent: self.coop_value_obs_convs(o_as[id_agent]['map'].permute(0, 3, 1, 2)) for id_agent in range(1, len(o_as))}

        extract_feature_map = torch.zeros(batch_size, self.graph_features, self.n_agents).to(device)
        extract_feature_map[:, :self.cnn_compression, 0] = greedy_cnn
        for id_agent in range(1, len(o_as)):
            extract_feature_map[:, :self.cnn_compression, id_agent] = coop_agents_cnn[id_agent]

        shared_feature = self.GFL(extract_feature_map)

        logits = torch.empty(batch_size, self.n_agents, 5).to(device)
        values = torch.empty(batch_size, self.n_agents).to(device)

        logits_inp = shared_feature[..., 0]
        if self.cfg['cnn_residual']:
            logits_inp = torch.cat([logits_inp, greedy_cnn], dim=1)
        logits[:, 0] = self.greedy_logits(logits_inp)
        if self.cfg['forward_values']:
            greedy_value_cnn = self.greedy_value_cnn(input_dict["obs"]["state"].permute(0, 3, 1, 2))
            coop_value_cnn = self.coop_value_cnn(input_dict["obs"]["state"].permute(0, 3, 1, 2))

            values[:, 0] = self.greedy_value_branch(torch.cat([greedy_value_obs_cnn, greedy_value_cnn], dim=1)).squeeze(1)

        for id_agent in range(1, len(o_as)):
            this_entity = shared_feature[..., id_agent]
            if self.cfg['cnn_residual']:
                this_entity = torch.cat([this_entity, coop_agents_cnn[id_agent]], dim=1)
            logits[:, id_agent] = self.coop_logits(this_entity)

            if self.cfg['forward_values']:
                value_cat = torch.cat([coop_value_cnn, coop_value_obs_cnn[id_agent]], dim=1)
                values[:, id_agent] = self.coop_value_branch(value_cat).squeeze(1)

        self._cur_value = values
        return logits.view(batch_size, self.n_agents*5), state

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

