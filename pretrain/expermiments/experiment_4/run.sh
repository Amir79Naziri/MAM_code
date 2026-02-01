#!/bin/bash

THIS_SCRIPT="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"

RESUME="${1}"


masking_strategy="cross-attention"
cross_attention_output_involevment="separate"
latent_dim=256
num_latents=128
n_lr=1e-4
olr=1e-5
gradient_clipping=1.0
batch_size=1
early_stopping_patience=5
cross_attn_regularization=1e-5
weight_decay=5e-4
accumulation_steps=5
num_epochs=10
cross_attention_heads=8
output_dir="/local/home/am/EX_M/scDAE_mn_4/"
advanced_masking=False


scripts_dir=/local/home/am/MscThesis/scDAE/pretrain/scripts

if [ "${RESUME}" = "TRUE" ]; then
  echo "Resuming training..."
  ${scripts_dir}/pretrain_resume_respai.sh \
    "${THIS_SCRIPT}/run.sh" \
    "${output_dir}" \
    "${masking_strategy}" \
    "${latent_dim}" \
    "${num_latents}" \
    "${n_lr}" \
    "${olr}" \
    "${gradient_clipping}" \
    "${batch_size}" \
    "${early_stopping_patience}" \
    "${cross_attn_regularization}" \
    "${weight_decay}" \
    "${accumulation_steps}" \
    "${num_epochs}" \
    "${cross_attention_heads}" \
    "${advanced_masking}" \
    "${cross_attention_output_involevment}"

else
  echo "Starting training..."
  ${scripts_dir}/pretrain_init_respai.sh \
    "${THIS_SCRIPT}/run.sh" \
    "${output_dir}" \
    "${masking_strategy}" \
    "${latent_dim}" \
    "${num_latents}" \
    "${n_lr}" \
    "${olr}" \
    "${gradient_clipping}" \
    "${batch_size}" \
    "${early_stopping_patience}" \
    "${cross_attn_regularization}" \
    "${weight_decay}" \
    "${accumulation_steps}" \
    "${num_epochs}" \
    "${cross_attention_heads}" \
    "${advanced_masking}" \
    "${cross_attention_output_involevment}"
fi

