import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Dropout


def clones(_to_clone_module, _clone_times, _is_deep=True):
    """Produce N identical layers."""
    copy_method = copy.deepcopy if _is_deep else copy.copy
    return nn.ModuleList(
        [copy_method(_to_clone_module) for _ in range(_clone_times if _is_deep else 1)]
    )


class PositionwiseFeedForward(nn.Module):
    def __init__(self, _dimensions, _feed_forward_dimensions, _dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(_dimensions, _feed_forward_dimensions)
        self.w_2 = nn.Linear(_feed_forward_dimensions, _dimensions)
        self.dropout = nn.Dropout(p=_dropout)

    def forward(self, _input_tensor):
        return self.w_2(self.dropout(F.relu(self.w_1(_input_tensor))))


class MultiHeadAttention(torch.jit.ScriptModule):
    def __init__(self, _multi_attention_heads, _dimensions, _dropout=0.1):
        """

        :param _multi_attention_heads: number of self attention head
        :param _dimensions: dimension of model
        :param _dropout:
        """
        super(MultiHeadAttention, self).__init__()

        assert _dimensions % _multi_attention_heads == 0
        # requires d_v = d_k, d_q = d_k = d_v = d_m / h
        self.d_k = int(_dimensions / _multi_attention_heads)
        self.h = _multi_attention_heads
        self.linears = clones(
            nn.Linear(_dimensions, _dimensions), 4
        )  # (q, k, v, last output layer)
        self.attention = None
        self.dropout = nn.Dropout(p=_dropout)

    # @torch.jit.script_method
    def dot_product_attention(self, _query, _key, _value, _mask):
        """
        Compute 'Scaled Dot Product Attention

        :param _query: (N, h, seq_len, d_q), h is multi-head
        :param _key: (N, h, seq_len, d_k)
        :param _value: (N, h, seq_len, d_v)
        :param _mask: None or (N, 1, seq_len, seq_len), 0 will be replaced with -1e9
        :return:
        """

        d_k = _value.size(-1)
        score = torch.matmul(_query, _key.transpose(-2, -1)) / math.sqrt(
            d_k
        )  # (N, h, seq_len, seq_len)
        if _mask is not None:
            score = score.masked_fill(
                _mask == 0, -1e9
            )  # score (N, h, seq_len, seq_len)
        p_attn = F.softmax(score, dim=-1)
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        return torch.matmul(p_attn, _value), p_attn

    # @torch.jit.script_method
    def forward(self, _query, _key, _value, _mask):
        batch_size = _query.size(0)

        # do all the linear projections in batch from d_model => h x d_k
        # (N, seq_len, d_m) -> (N, seq_len, h, d_k) -> (N, h, seq_len, d_k)
        _query, _key, _value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (_query, _key, _value))
        ]

        # apply attention on all the projected vectors in batch.
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        product_and_attention = self.dot_product_attention(
            _query, _key, _value, _mask=_mask
        )
        x = product_and_attention[0]
        # self.attention = self.dropout(product_and_attention[1])

        # "Concat" using a view and apply a final linear.
        # (N, seq_len, d_m)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # (N, seq_len, d_m)
        return self.linears[-1](x)


class MasterDecoder(nn.Module):
    def __init__(
        self, _multi_heads_count, _dimensions, _stacks, _feed_forward_size, weights=None
    ):
        super(MasterDecoder, self).__init__()
        _dropout = 0.1
        self.layer_num = _stacks
        self.head_num = _multi_heads_count
        self.hidden_dim = _dimensions
        self.weights = weights

        assert _dimensions % _multi_heads_count == 0
        self.size_per_head = _dimensions // _multi_heads_count

        self.attention = nn.ModuleList(
            [
                MultiHeadAttention(_multi_heads_count, _dimensions, _dropout)
                for _ in range(_stacks)
            ]
        )
        self.source_attention = nn.ModuleList(
            [
                MultiHeadAttention(_multi_heads_count, _dimensions, _dropout)
                for _ in range(_stacks)
            ]
        )
        self.position_feed_forward = nn.ModuleList(
            [
                PositionwiseFeedForward(_dimensions, _feed_forward_size, _dropout)
                for _ in range(_stacks)
            ]
        )
        self.dropout = Dropout(_dropout)
        self.layer_norm = torch.nn.LayerNorm(_dimensions, eps=1e-6)

        if self.weights:
            self.load_weights()

    def load_weights(self):
        def cvt_weight(d):
            return d.transpose(-1, -2).contiguous()

        # 所有层共享 layer_norm
        self.layer_norm.weight.data = self.weights[0][0]
        self.layer_norm.bias.data = self.weights[0][1]

        for i in range(self.layer_num):
            w = self.weights[i]
            # self_attn
            self.attention[i].linears[0].weight.data = cvt_weight(w[2])
            self.attention[i].linears[1].weight.data = cvt_weight(w[3])
            self.attention[i].linears[2].weight.data = cvt_weight(w[4])

            self.attention[i].linears[0].bias.data = w[5]
            self.attention[i].linears[1].bias.data = w[6]
            self.attention[i].linears[2].bias.data = w[7]

            self.attention[i].linears[3].weight.data = cvt_weight(w[8])
            self.attention[i].linears[3].bias.data = w[9]

            # cross attn
            self.source_attention[i].linears[0].weight.data = cvt_weight(w[12])
            self.source_attention[i].linears[1].weight.data = cvt_weight(w[13])
            self.source_attention[i].linears[2].weight.data = cvt_weight(w[14])

            self.source_attention[i].linears[0].bias.data = w[15]
            self.source_attention[i].linears[1].bias.data = w[16]
            self.source_attention[i].linears[2].bias.data = w[17]

            self.source_attention[i].linears[3].weight.data = cvt_weight(w[18])
            self.source_attention[i].linears[3].bias.data = w[19]

            # FFN
            self.position_feed_forward[i].w_1.weight.data = cvt_weight(w[22])
            self.position_feed_forward[i].w_1.bias.data = w[23]
            self.position_feed_forward[i].w_2.weight.data = cvt_weight(w[24])
            self.position_feed_forward[i].w_2.bias.data = w[25]

    def forward(self, target, memory, source_mask, target_mask, step):
        """
        source_mask: memory 的 mask
        """
        inputs = target
        for i in range(self.layer_num):
            # self_attention MaskedMHA
            input_norm = self.layer_norm(inputs)
            query = inputs + self.attention[i](
                input_norm, input_norm, input_norm, target_mask
            )

            # MHA
            query_norm = self.layer_norm(query)
            query = query + self.source_attention[i](
                query_norm, memory, memory, source_mask
            )

            input_norm = self.layer_norm(query)
            inputs = query + self.position_feed_forward[i](input_norm)

        return inputs

    def get_master_decoder_weights(self, gpu=True):
        weights = []

        def cvt_weight(fc):
            return copy.deepcopy(fc.weight.data.transpose(-1, -2).contiguous())

        for i in range(self.layer_num):
            self_attention = self.attention[i]
            cross_attention = self.source_attention[i]
            ffn = self.position_feed_forward[i]
            layer_weights = [
                # self_attention
                copy.deepcopy(self.layer_norm.weight.data),  # 0
                copy.deepcopy(self.layer_norm.bias.data),  # 1
                cvt_weight(self_attention.linears[0]),
                cvt_weight(self_attention.linears[1]),
                cvt_weight(self_attention.linears[2]),
                copy.deepcopy(self_attention.linears[0].bias.data),
                copy.deepcopy(self_attention.linears[1].bias.data),
                copy.deepcopy(self_attention.linears[2].bias.data),
                cvt_weight(self_attention.linears[3]),
                copy.deepcopy(self_attention.linears[3].bias.data),
                # cross_attention
                copy.deepcopy(self.layer_norm.weight.data),  # 10
                copy.deepcopy(self.layer_norm.bias.data),  # 11
                cvt_weight(cross_attention.linears[0]),
                cvt_weight(cross_attention.linears[1]),
                cvt_weight(cross_attention.linears[2]),
                copy.deepcopy(cross_attention.linears[0].bias.data),
                copy.deepcopy(cross_attention.linears[1].bias.data),
                copy.deepcopy(cross_attention.linears[2].bias.data),
                cvt_weight(cross_attention.linears[3]),
                copy.deepcopy(cross_attention.linears[3].bias.data),
                # ffn
                copy.deepcopy(self.layer_norm.weight.data),  # 20
                copy.deepcopy(self.layer_norm.bias.data),  # 21
                cvt_weight(ffn.w_1),
                copy.deepcopy(ffn.w_1.bias.data),
                cvt_weight(ffn.w_2),
                copy.deepcopy(ffn.w_2.bias.data),
            ]
            weights.append(layer_weights)

        layer_norm_weight = self.layer_norm.weight.data
        layer_norm_bias = self.layer_norm.bias.data
        for w in weights:
            assert w[0].equal(layer_norm_weight)
            assert w[1].equal(layer_norm_bias)
            assert w[10].equal(layer_norm_weight)
            assert w[11].equal(layer_norm_bias)
            assert w[20].equal(layer_norm_weight)
            assert w[21].equal(layer_norm_bias)

        if gpu:
            for i in range(len(weights)):
                for j in range(len(weights[0])):
                    weights[i][j] = weights[i][j].cuda()

        return weights
