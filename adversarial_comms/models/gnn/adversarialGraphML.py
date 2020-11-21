import math
import numpy as np
import torch
import torch.nn as nn

from .graphML import GraphFilterBatchGSO

def batchLSIGFA(h0, h1, N0, SK, x, bias=None, aggregation=lambda y, dim: torch.sum(y, dim=dim)):
    """
    batchLSIGF(filter_taps, GSO_K, input, bias=None) Computes the output of a
        linear shift-invariant graph filter on input and then adds bias.

    In this case, we consider that there is a separate GSO to be used for each
    of the signals in the batch. In other words, SK[b] is applied when filtering
    x[b] as opposed to applying the same SK to all the graph signals in the
    batch.

    Inputs:
        filter_taps: vector of filter taps; size:
            output_features x edge_features x filter_taps x input_features
        GSO_K: collection of matrices; size:
            batch_size x edge_features x filter_taps x number_nodes x number_nodes
        input: input signal; size:
            batch_size x input_features x number_nodes
        bias: size: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; size:
            batch_size x output_features x number_nodes
    """
    # Get the parameter numbers:
    assert h0.shape == h1.shape
    F = h0.shape[0]
    E = h0.shape[1]
    K = h0.shape[2]
    G = h0.shape[3]
    B = SK.shape[0]
    assert SK.shape[1] == E
    assert SK.shape[2] == K
    N = SK.shape[3]
    assert SK.shape[4] == N
    assert x.shape[0] == B
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation I've been using:
    # h in F x E x K x G
    # SK in B x E x K x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N
    SK = SK.permute(1, 2, 0, 3, 4)
    # Now, SK is of shape E x K x B x N x N so that we can multiply by x of
    # size B x G x N to get
    z = torch.matmul(x, SK)
    # which is of size E x K x B x G x N
    # Now, we have already carried out the multiplication across the dimension
    # of the nodes. Now we need to focus on the K, F, G.
    # Let's start by putting B and N in the front
    z = z.permute(1, 2, 4, 0, 3).reshape([K, B, N, E * G])
    # so that we get z in B x N x EKG.
    # Now adjust the filter taps so they are of the form EKG x F
    h0 = h0.permute(2, 1, 3, 0).reshape([K, G * E, F])
    h1 = h1.permute(2, 1, 3, 0).reshape([K, G * E, F])
    #h1 = h1.reshape([F, G * E * K]).permute(1, 0)
    # Multiply
    if N0 == 0:
        y = torch.empty(K, B, N, G * E).to(z.device)
        for k in range(K):
            y[k] = torch.matmul(z[k], h1[k])
        y = aggregation(y, 0)
        # to get a result of size B x N x F. And permute
        y = y.permute(0, 2, 1)
    else:
        z0 = z[:, :, :N0]
        z1 = z[:, :, N0:]
        y0 = torch.empty(K, B, N0, G * E).to(z.device)
        y1 = torch.empty(K, B, N-N0, G * E).to(z.device)
        for k in range(K):
            y0[k] = torch.matmul(z0[k], h0[k])
            y1[k] = torch.matmul(z1[k], h1[k])
        y0 = aggregation(y0, 0)
        y1 = aggregation(y1, 0)
        # to get a result of size B x N x F. And permute
        y0 = y0.permute(0, 2, 1)
        y1 = y1.permute(0, 2, 1)
        y = torch.cat([y0, y1], dim = 2) # concat along N
    # to get it back in the right order: B x F x N.
    # Now, in this case, each element x[b,:,:] has adequately been filtered by
    # the GSO S[b,:,:,:]
    if bias is not None:
        y = y + bias
    return y

class GraphFilterBatchGSOA(GraphFilterBatchGSO):
    def __init__(self, G, F, K, N0, E = 1, bias = True, aggregation='sum'):
        super().__init__(G, F, K, E, bias)
        self.weight0 = self.weight
        self.weight1 = nn.parameter.Parameter(torch.Tensor(self.F, self.E, self.K, self.G))
        self.N0 = N0
        self.reset_parameters()
        self.aggregation = {
            "sum": lambda y, dim: torch.sum(y, dim=dim),
            "median": lambda y, dim: torch.median(y, dim=dim)[0],
            "min": lambda y, dim: torch.min(y, dim=dim)[0]
        }[aggregation]

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'weight1'):
            stdv = 1. / math.sqrt(self.G * self.K)
            self.weight1.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.forward_gpvae(x) if self.K == 2 else batchLSIGFA(self.weight0, self.weight1, self.N0, self.SK, x, self.bias, aggregation=self.aggregation)

    def forward_gpvae(self, x):
        # K=1
        hx_0_0 = torch.matmul(self.weight0[:, 0, 0, :], x[:, :, :self.N0])
        hx_0_1 = torch.matmul(self.weight1[:, 0, 0, :], x[:, :, self.N0:])
        hx_0 = torch.cat([hx_0_0, hx_0_1], dim=2)

        # K=2
        neighbors = self.aggregation(x[:, :, :, None] * self.S, dim=2)
        hx_1_0 = torch.matmul(self.weight0[:, 0, 1, :], neighbors[:, :, :self.N0])
        hx_1_1 = torch.matmul(self.weight1[:, 0, 1, :], neighbors[:, :, self.N0:])
        hx_1 = torch.cat([hx_1_0, hx_1_1], dim=2)

        output = hx_0 + hx_1
        return output

    def forward_naive(self, x):
        bs, features, n_agents = x.shape
        output = torch.zeros(bs, features, n_agents)
        for b in range(bs):
            sxas = torch.zeros(self.K, features, n_agents)
            sk = torch.eye(n_agents).expand(n_agents, n_agents)
            for k in range(self.K):
                sx = torch.matmul(x[b], sk)
                h0 = self.weight0[:, 0, k, :]
                h1 = self.weight1[:, 0, k, :]
                if self.N0 == 0:
                    sxas[k] = torch.matmul(h1, sx)
                else:
                    sxa0 = torch.matmul(h0, sx[:, :self.N0])
                    sxa1 = torch.matmul(h1, sx[:, self.N0:])
                    sxas[k] = torch.cat([sxa0, sxa1], dim=1)  # concat along N
                sk = torch.matmul(self.S[b, 0], sk)

            output[b] = self.aggregation(sxas, 0)
        return output
