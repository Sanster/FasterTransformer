import torch

USE_CACHE_BATCH_MAJOR_ATTENTION = False


def get_op_cache_config(size_per_head, is_fp16):
    x = 8 if is_fp16 else 4
    use_batch_major_op_cache = (
        True
        if USE_CACHE_BATCH_MAJOR_ATTENTION == True and size_per_head % x == 0
        else False
    )
    x = x if use_batch_major_op_cache else 1
    return use_batch_major_op_cache, x


class FTDecoder(torch.nn.Module):
    def __init__(
        self,
        layer_num,
        head_num,
        head_size,
        mem_hidden_dim,
        weights,
        is_fp16,
        path="./lib/libpyt_fastertransformer.so",
    ):
        super().__init__()
        self.layer_num = layer_num
        self.hidden_dim = head_num * head_size
        self.head_num = head_num
        self.head_size = head_size
        self.fp16 = is_fp16
        self.decoders = []
        torch.classes.load_library(path)

        for i in range(layer_num):
            self.decoders.append(
                torch.classes.FasterTransformer.Decoder(
                    head_num, head_size, mem_hidden_dim, *weights[i]
                )
            )

    def to_cuda(self):
        for i in range(self.layer_num):
            for j in range(len(self.weights[i])):
                self.weights[i][j] = self.weights[i][j].cuda()

    def forward(self, inputs, memory, memory_seq_lens, self_cache, mem_cache, step):
        dtype = torch.half if self.fp16 else torch.float32
        use_batch_major_op_cache, _ = get_op_cache_config(self.head_size, self.fp16)
        if use_batch_major_op_cache == False:
            self_cache_tmp = [
                torch.zeros(
                    self.layer_num,
                    1,
                    self_cache[0].size(2),
                    self.hidden_dim,
                    dtype=dtype,
                    device="cuda",
                ),
                torch.zeros(
                    self.layer_num,
                    1,
                    self_cache[1].size(2),
                    self.hidden_dim,
                    dtype=dtype,
                    device="cuda",
                ),
            ]
            self_cache[0] = torch.cat([self_cache[0], self_cache_tmp[0]], 1)
            self_cache[1] = torch.cat([self_cache[1], self_cache_tmp[1]], 1)

        output = inputs
        for i in range(self.layer_num):
            output = self.decoders[i].forward(
                output,
                memory,
                memory_seq_lens,
                (self_cache[0][i], self_cache[1][i]),
                mem_cache[i],
                step,
            )
        return output, self_cache, mem_cache
