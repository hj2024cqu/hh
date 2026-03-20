#!/bin/bash

# row_block_size=32
python test.py /home/hej/model/float/stage1_pruned \
    --mantissa_bits 4 \
    --row_block_size 32 \
    --output_dir /home/hej/model/float/sparsegpt/compressed_32_4bits/mantissa_4bit

# row_block_size=64
python test.py /home/hej/model/float/stage1_pruned \
    --mantissa_bits 4 \
    --row_block_size 64 \
    --output_dir /home/hej/model/float/sparsegpt/compressed_64_4bits/mantissa_4bit