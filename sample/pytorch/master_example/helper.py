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


def init_op_cache(
    layer_num,
    batch_size,
    beam_width,
    max_seq_len,
    decoding_max_seq_len,
    head_num,
    size_per_head,
    hidden_dim,
    is_fp16,
):
    use_batch_major_op_cache, x = get_op_cache_config(size_per_head, is_fp16)
    dtype = torch.half if is_fp16 else torch.float32
    if use_batch_major_op_cache == True:
        self_cache = [
            torch.zeros(
                layer_num,
                batch_size * beam_width,
                head_num,
                size_per_head // x,
                decoding_max_seq_len,
                x,
                dtype=dtype,
                device="cuda",
            ),
            torch.zeros(
                layer_num,
                batch_size * beam_width,
                head_num,
                decoding_max_seq_len,
                size_per_head,
                dtype=dtype,
                device="cuda",
            ),
        ]
    else:
        self_cache = [
            torch.zeros(
                layer_num,
                0,
                batch_size * beam_width,
                hidden_dim,
                dtype=dtype,
                device="cuda",
            ),
            torch.zeros(
                layer_num,
                0,
                batch_size * beam_width,
                hidden_dim,
                dtype=dtype,
                device="cuda",
            ),
        ]

    # always use old format for cross attention for now
    mem_cache = torch.zeros(
        layer_num,
        2,
        batch_size * beam_width,
        max_seq_len,
        hidden_dim,
        dtype=dtype,
        device="cuda",
    )

    return self_cache, mem_cache


def init_onmt_cache(layer_num, memory_bank):
    cache = {}
    for i in range(layer_num):
        layer_cache = {"memory_keys": None, "memory_values": None}
        layer_cache["self_keys"] = None
        layer_cache["self_values"] = None
        cache[i] = layer_cache
    return cache
