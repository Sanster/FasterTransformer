import copy

import torch


class DecoderWeights(object):
    def __init__(self, layer_num, hidden_dim, d_ffn, same_layer_norm=True):
        self.layer_num = layer_num
        self.w = [[] for _ in range(layer_num)]

        layernorm_gamma = torch.zeros(hidden_dim)
        layernorm_beta = torch.zeros(hidden_dim)
        torch.nn.init.uniform_(layernorm_gamma, -1, 1)
        torch.nn.init.uniform_(layernorm_beta, -1, 1)

        for layer_weights in self.w:
            if same_layer_norm:
                layer_weights.append(layernorm_gamma)  # self_layernorm_gamma
                layer_weights.append(layernorm_beta)  # self_layernorm_beta
            else:
                layer_weights.append(torch.zeros(hidden_dim))  # self_layernorm_gamma
                layer_weights.append(torch.zeros(hidden_dim))  # self_layernorm_beta

            layer_weights.append(torch.zeros(hidden_dim, hidden_dim))  # self_kernel_q
            layer_weights.append(torch.zeros(hidden_dim, hidden_dim))  # self_kernel_k
            layer_weights.append(torch.zeros(hidden_dim, hidden_dim))  # self_kernel_v
            layer_weights.append(torch.zeros(hidden_dim))  # self_bias_q
            layer_weights.append(torch.zeros(hidden_dim))  # self_bias_k
            layer_weights.append(torch.zeros(hidden_dim))  # self_bias_v
            layer_weights.append(
                torch.zeros(hidden_dim, hidden_dim)
            )  # self_output_kernel
            layer_weights.append(torch.zeros(hidden_dim))  # self_output_bias
            layer_weights.append(torch.zeros(hidden_dim))  # cross_layernorm_gamma
            layer_weights.append(torch.zeros(hidden_dim))  # cross_layernorm_beta
            layer_weights.append(torch.zeros(hidden_dim, hidden_dim))  # cross_kernel_q
            layer_weights.append(torch.zeros(hidden_dim, hidden_dim))  # cross_kernel_k
            layer_weights.append(torch.zeros(hidden_dim, hidden_dim))  # cross_kernel_v
            layer_weights.append(torch.zeros(hidden_dim))  # cross_bias_q
            layer_weights.append(torch.zeros(hidden_dim))  # cross_bias_k
            layer_weights.append(torch.zeros(hidden_dim))  # cross_bias_v
            layer_weights.append(
                torch.zeros(hidden_dim, hidden_dim)
            )  # cross_output_kernel
            layer_weights.append(torch.zeros(hidden_dim))  # cross_output_bias
            layer_weights.append(torch.zeros(hidden_dim))  # ffn_layernorm_gamma
            layer_weights.append(torch.zeros(hidden_dim))  # ffn_layernorm_beta
            layer_weights.append(torch.zeros(hidden_dim, d_ffn))  # inter_kernel
            layer_weights.append(torch.zeros(d_ffn))  # inter_bias
            layer_weights.append(torch.zeros(d_ffn, hidden_dim))  # output_kernel
            layer_weights.append(torch.zeros(hidden_dim))  # output_bias

            if same_layer_norm:
                for i in range(2, len(layer_weights)):
                    torch.nn.init.uniform_(layer_weights[i], -1, 1)
                layer_weights[10] = copy.deepcopy(layernorm_gamma)
                layer_weights[11] = copy.deepcopy(layernorm_beta)

                layer_weights[20] = copy.deepcopy(layernorm_gamma)
                layer_weights[21] = copy.deepcopy(layernorm_beta)
            else:
                for i in range(len(layer_weights)):
                    torch.nn.init.uniform_(layer_weights[i], -1, 1)

    def to_cuda(self):
        for i in range(self.layer_num):
            for j in range(len(self.w[i])):
                self.w[i][j] = self.w[i][j].cuda()
