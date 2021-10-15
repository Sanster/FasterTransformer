import torch
from onmt.decoders.transformer import TransformerDecoderLayer


class ONMTDecoder(torch.nn.Module):
    def __init__(self, layer_num, head_num, head_size, d_ff, weights):
        super().__init__()
        self.layer_num = layer_num
        self.hidden_dim = head_num * head_size
        self.decoders = torch.nn.ModuleList()
        for i in range(layer_num):
            self.decoders.append(
                TransformerDecoderLayer(self.hidden_dim, head_num, d_ff, 0, 0)
            )

        try:
            weights = weights.w
        except:
            pass

        def cvt_weight(data):
            return data.transpose(-1, -2).contiguous()

        for i in range(layer_num):
            # fmt: off
            self.decoders[i].layer_norm_1.weight.data = weights[i][0]
            self.decoders[i].layer_norm_1.bias.data = weights[i][1]
            self.decoders[i].self_attn.linear_query.weight.data = cvt_weight(weights[i][2])
            self.decoders[i].self_attn.linear_keys.weight.data = cvt_weight(weights[i][3])
            self.decoders[i].self_attn.linear_values.weight.data = cvt_weight(weights[i][4])
            self.decoders[i].self_attn.linear_query.bias.data = weights[i][5]
            self.decoders[i].self_attn.linear_keys.bias.data = weights[i][6]
            self.decoders[i].self_attn.linear_values.bias.data = weights[i][7]
            self.decoders[i].self_attn.final_linear.weight.data = cvt_weight(weights[i][8])
            self.decoders[i].self_attn.final_linear.bias.data = weights[i][9]
            self.decoders[i].layer_norm_2.weight.data = weights[i][10]
            self.decoders[i].layer_norm_2.bias.data = weights[i][11]

            self.decoders[i].context_attn.linear_query.weight.data = cvt_weight(weights[i][12])
            self.decoders[i].context_attn.linear_keys.weight.data = cvt_weight(weights[i][13])
            self.decoders[i].context_attn.linear_values.weight.data = cvt_weight(weights[i][14])
            self.decoders[i].context_attn.linear_query.bias.data = weights[i][15]
            self.decoders[i].context_attn.linear_keys.bias.data = weights[i][16]
            self.decoders[i].context_attn.linear_values.bias.data = weights[i][17]
            self.decoders[i].context_attn.final_linear.weight.data = cvt_weight(weights[i][18])
            self.decoders[i].context_attn.final_linear.bias.data = weights[i][19]

            self.decoders[i].feed_forward.layer_norm.weight.data = weights[i][20]
            self.decoders[i].feed_forward.layer_norm.bias.data = weights[i][21]

            self.decoders[i].feed_forward.w_1.weight.data = cvt_weight(weights[i][22])
            self.decoders[i].feed_forward.w_1.bias.data = weights[i][23]
            self.decoders[i].feed_forward.w_2.weight.data = cvt_weight(weights[i][24])
            self.decoders[i].feed_forward.w_2.bias.data = weights[i][25]
            # fmt: on

    def forward(self, inputs, memory, src_pad_msk, cache, step):
        output = inputs
        for i in range(self.layer_num):
            output, _, _ = self.decoders[i](
                output, memory, src_pad_msk, None, cache[i], step
            )
        return output
