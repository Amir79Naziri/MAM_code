#!/bin/bash

THIS_SCRIPT="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"

task="test"
masking_strategy="cross-attention"
cross_attention_output_involevment="encoder"
latent_dim=256
num_latents=128
lr=1e-4
gradient_clipping=1.0
batch_size=4
early_stopping_patience=5
cross_attn_regularization=1e-5
weight_decay=5e-4
accumulation_steps=5
num_epochs=10
cross_attention_heads=8
output_dir="/local/home/am/EX_M/scDAE_mn_f_6/"
pretrained_model_dir="/local/home/am/EX_M/scDAE_mn_6/best/model.ckpt"
advanced_masking=True
frozen_masking_layers=False
linear_probe=False

scripts_dir=/local/home/am/MscThesis/scDAE/finetune/scripts

echo "Starting testing..."
${scripts_dir}/test_respai.sh \
"${THIS_SCRIPT}/run.sh" \
"${output_dir}" \
"${masking_strategy}" \
"${latent_dim}" \
"${num_latents}" \
"${lr}" \
"${gradient_clipping}" \
"${batch_size}" \
"${early_stopping_patience}" \
"${cross_attn_regularization}" \
"${weight_decay}" \
"${accumulation_steps}" \
"${num_epochs}" \
"${cross_attention_heads}" \
"${advanced_masking}" \
"${cross_attention_output_involevment}" \
"${pretrained_model_dir}" \
"${task}" \
"${frozen_masking_layers}" \
"${linear_probe}"


