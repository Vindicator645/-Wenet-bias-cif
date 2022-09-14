#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
#           2022 Binbin Zhang(binbizha@qq.com)

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

stage=6 # start from 0 if you need to start from data preparation
stop_stage=6

num_nodes=1

dir=exp/rnnt
checkpoint=

# use average_checkpoint will get better result
. tools/parse_options.sh || exit 1;



if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/final.pt \
    --output_file $dir/final.zip \
    --output_quant_file $dir/final_quant.zip
fi


