TL;DR: When use FasterTransformer hidden size of PositionwiseFeedForward must equal to 4*hidden_dim

Different between MASTERDecoder and ONMTDecoder
1. MASTERDecoder not use cache
2. MASTERDecoder use same layer_norm in all layers


```bash
pip3 install tabulate

#batch_size
#beam_width
#head_number
#size_per_head
#vocab_size
#sequence_length
#encoder_hidden_dim
#is_use_fp16

./bin/decoding_gemm 1 1 4 128 7001 32 512 0
```

- hidden_dim: head_num * head_size
- d_ffn: hidden size of PositionwiseFeedForward
- only tested in batch_size==1

## d_ffn != 4*hidden_dim

```bash
python pytorch/master_example/master_example.py \
--batch_size=1 \
--layer_num=2 \
--seq_len=32 \
--head_num=4 \
--head_size=128 \
--d_ffn=512 \
--step=4 \
--run_ft

  step  diff name      max relative diff
------  -----------  -------------------
     0  ft-master           53.7376
     0  ft-onmt             53.7376
     0  onmt-master          0.000292255
     1  ft-master          103.101
     1  ft-onmt            103.101
     1  onmt-master          0.000151085
     2  ft-master          209.367
     2  ft-onmt            209.367
     2  onmt-master          0.000504965
     3  ft-master          145.746
     3  ft-onmt            145.746
     3  onmt-master          0.00112149
```



## d_ffn == 4*hidden_dim

```bash
python pytorch/master_example/master_example.py \
--batch_size=1 \
--layer_num=2 \
--seq_len=32 \
--head_num=4 \
--head_size=128 \
--d_ffn=2048 \
--step=4 \
--run_ft

  step  diff name      max relative diff
------  -----------  -------------------
     0  ft-master            7.16644e-05
     0  ft-onmt              4.15602e-05
     0  onmt-master          3.26929e-05
     1  ft-master            0.000143105
     1  ft-onmt              0.000181676
     1  onmt-master          3.85782e-05
     2  ft-master            0.000124314
     2  ft-onmt              0.000559099
     2  onmt-master          0.00060457
     3  ft-master            9.97579e-05
     3  ft-onmt              0.000107287
     3  onmt-master          6.91183e-05
```
