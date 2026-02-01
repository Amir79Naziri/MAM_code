#!/bin/bash



INPUT_PARENT_SCRIPT="${1}"
output_dir=${2}
masking_strategy=${3}
latent_dim=${4}
num_latents=${5}
nlr=${6}
olr=${7}
gradient_clipping=${8}
batch_size=${9}
early_stopping_patience=${10}
cross_attn_regularization=${11}
weight_decay=${12}
accumulation_steps=${13}
num_epochs=${14}
cross_attention_heads=${15}
advanced_masking=${16}
cross_attention_output_involevment=${17}

echo "Output directory: ${output_dir}"
echo "Masking strategy: ${masking_strategy}"
echo "Latent dim: ${latent_dim}"
echo "Num latents: ${num_latents}"
echo "NLr: ${nlr}"
echo "OLr: ${olr}"
echo "Gradient clipping: ${gradient_clipping}"
echo "Batch size: ${batch_size}"
echo "Early stopping patience: ${early_stopping_patience}"
echo "Cross attn regularization: ${cross_attn_regularization}"
echo "Weight decay: ${weight_decay}"
echo "Accumulation steps: ${accumulation_steps}"
echo "Num epochs: ${num_epochs}"
echo "Cross attention heads: ${cross_attention_heads}"
echo "Advanced masking: ${advanced_masking}"
echo "Cross attention output involvement: ${cross_attention_output_involevment}"


RESUME="TRUE"
export CUDA_VISIBLE_DEVICES=1,3
activate base 
torchrun --standalone --nproc_per_node=2 ../main.py \
    --masking_strategy ${masking_strategy} \
    --latent_dim ${latent_dim} \
    --num_latents ${num_latents} \
    --nlr ${nlr} \
    --olr ${olr} \
    --gradient_clipping ${gradient_clipping} \
    --batch_size ${batch_size} \
    --early_stopping_patience ${early_stopping_patience} \
    --cross_attn_regularization ${cross_attn_regularization} \
    --weight_decay ${weight_decay} \
    --accumulation_steps ${accumulation_steps} \
    --num_epochs ${num_epochs} \
    --cross_attention_heads ${cross_attention_heads} \
    --advanced_masking ${advanced_masking} \
    --cross_attention_output_involevment ${cross_attention_output_involevment} \
    --train_data /local/home/am/EX_M/datasets/scRNA/pbmc68k/seed_32/scFoundation/train/pbmc68k_train_processed.h5ad \
    --val_data /local/home/am/EX_M/datasets/scRNA/pbmc68k/seed_32/scFoundation/validation/pbmc68k_val_processed.h5ad \
    --output_dir ${output_dir} \
    --initialized_model_dir /local/home/am/EX_M/scFoundation/pretrained_model/model.ckpt \
    --resume True \
    &

trap " ${INPUT_PARENT_SCRIPT} \"${RESUME}\" " USR1

wait