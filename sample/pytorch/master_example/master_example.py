from __future__ import print_function
import argparse
import os
from collections import defaultdict

import torch

torch.manual_seed(42)


from tabulate import tabulate
from onmt.utils.misc import sequence_mask

from helper import init_op_cache, init_onmt_cache
from ft_decoder import FTDecoder
from onmt_decoder import ONMTDecoder
from master_decoder import MasterDecoder
from decoder_weights import DecoderWeights


def generate_master_target_mask(batch_size, seq_len, current_step):
    target_sub_mask = torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.uint8)
    ).cuda()
    target_pad_mask = torch.BoolTensor(seq_len).cuda() * False
    target_pad_mask[0 : current_step + 1] = True
    target_pad_mask = target_pad_mask.reshape(batch_size, 1, seq_len, 1)
    target_mask = target_pad_mask & target_sub_mask.bool()
    return target_mask


def prepare_master_outputs_and_source_mask(
    inp, mem_seq_lens, batch_size, seq_len, hidden_dim
):
    # init master input/output
    # [batch_size, 1, hidden_dim]
    output_master = torch.zeros((batch_size, seq_len, hidden_dim)).cuda()
    output_master[:, 0, :] = inp.clone()

    master_source_mask = torch.zeros((seq_len, seq_len), dtype=torch.uint8).cuda()
    master_source_mask[:, 0 : mem_seq_lens[0]] = 1
    master_source_mask = master_source_mask.bool()
    return output_master, master_source_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--layer_num", type=int, help="number of layers")
    parser.add_argument("--seq_len", type=int, help="sequence length")
    parser.add_argument("--head_num", type=int, help="head number")
    parser.add_argument("--head_size", type=int, help="size per head")
    parser.add_argument("--d_ffn", type=int, help="FFN hidden size")
    parser.add_argument("--run_ft", action="store_true")
    parser.add_argument(
        "--use_decoder_weights",
        action="store_true",
        help="Use DecoderWeights to create weights",
    )
    parser.add_argument("--step", type=int, default=4, help="decoding step number")
    parser.add_argument(
        "--ths_path",
        type=str,
        default="./lib/libpyt_fastertransformer.so",
        help="path of the pyt_fastertransformer dynamic lib file",
    )

    args = parser.parse_args()

    hidden_dim = args.head_num * args.head_size
    print(f"hidden_dim: {hidden_dim} (args.head_num * args.head_size)")

    inp = torch.empty(args.batch_size, 1, hidden_dim).cuda()
    mem = torch.empty(
        args.batch_size, args.seq_len, hidden_dim
    ).cuda()  # We assume mem_hidden_dim = hidden_dim
    torch.nn.init.uniform_(inp, -1, 1)
    torch.nn.init.uniform_(mem, -1, 1)
    mem_seq_lens = torch.randint(
        1, args.seq_len + 1, (args.batch_size,), dtype=torch.int32
    ).cuda()
    src_pad_mask = ~sequence_mask(mem_seq_lens, args.seq_len).unsqueeze(1)

    if args.use_decoder_weights:
        w = DecoderWeights(args.layer_num, hidden_dim, args.d_ffn)
        w.to_cuda()
        weights = w.w

        master_decoder = MasterDecoder(
            _stacks=args.layer_num,
            _multi_heads_count=args.head_num,
            _dimensions=args.head_num * args.head_size,
            _feed_forward_size=args.d_ffn,
            weights=weights,
        )
    else:
        master_decoder = MasterDecoder(
            _stacks=args.layer_num,
            _multi_heads_count=args.head_num,
            _dimensions=args.head_num * args.head_size,
            _feed_forward_size=args.d_ffn,
        )
        weights = master_decoder.get_master_decoder_weights()

    master_decoder = master_decoder.cuda()
    master_decoder.eval()

    onmt_decoder = ONMTDecoder(
        args.layer_num, args.head_num, args.head_size, args.d_ffn, weights
    )
    onmt_decoder = onmt_decoder.cuda()
    onmt_decoder.eval()

    with torch.no_grad():
        if args.run_ft:
            ft_decoder = FTDecoder(
                args.layer_num,
                args.head_num,
                args.head_size,
                hidden_dim,
                weights,
                is_fp16=False,
                path=os.path.abspath(args.ths_path),
            )

            self_cache, mem_cache = init_op_cache(
                args.layer_num,
                args.batch_size,
                1,
                args.seq_len,
                args.seq_len,
                args.head_num,
                args.head_size,
                hidden_dim,
                is_fp16=False,
            )
            output_ft = inp.clone()

        cache = init_onmt_cache(args.layer_num, mem)
        output_onmt = inp.clone()

        output_master, master_source_mask = prepare_master_outputs_and_source_mask(
            inp, mem_seq_lens, args.batch_size, args.seq_len, hidden_dim
        )

        print(f"mem_seq_lens: {mem_seq_lens.shape}, value: {mem_seq_lens}")

        table_data = defaultdict(list)

        for i in range(args.step):
            # ONMT Forward
            output_onmt = onmt_decoder(output_onmt, mem, src_pad_mask, cache, 0)

            ####### Run MASTER Decoder ######
            master_target_mask = generate_master_target_mask(
                args.batch_size, args.seq_len, i
            )
            output_master_new = master_decoder(
                output_master,
                mem,
                source_mask=master_source_mask,
                target_mask=master_target_mask,
                step=i,
            )
            output_master[:, i + 1, :] = output_master_new[:, i : i + 1, :]
            ##############################

            if args.run_ft:
                # FasterTransformer Forward
                output_ft, self_cache, mem_cache = ft_decoder(
                    output_ft, mem, mem_seq_lens, self_cache, mem_cache, i
                )

                diff = torch.abs((output_ft - output_master[:, i + 1, :]) / output_ft)
                table_data["step"].append(i)
                table_data["diff name"].append("ft-master")
                table_data["max relative diff"].append(torch.max(diff))

                diff = torch.abs((output_ft - output_onmt) / output_ft)
                table_data["step"].append(i)
                table_data["diff name"].append("ft-onmt")
                table_data["max relative diff"].append(torch.max(diff))

            diff = torch.abs((output_master[:, i + 1, :] - output_onmt) / output_onmt)
            table_data["step"].append(i)
            table_data["diff name"].append("onmt-master")
            table_data["max relative diff"].append(torch.max(diff))

    print(tabulate(table_data, headers="keys"))


if __name__ == "__main__":
    main()
