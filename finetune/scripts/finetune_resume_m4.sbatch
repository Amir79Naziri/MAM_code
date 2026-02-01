#!/bin/bash
#SBATCH --job-name=finetuning
#SBATCH --output=slurm-finetuning-%j.out
#SBATCH --error=slurm-finetuning-%j.err
#SBATCH --gres=gpu:a40:4
#SBATCH --qos=m4
#SBATCH --time=2:00:00
#SBATCH --signal=B:USR1@120
#SBATCH -c 10
#SBATCH --mem=80G


INPUT_PARENT_SCRIPT="${1}"
output_dir=${2}
masking_strategy=${3}
latent_dim=${4}
num_latents=${5}
lr=${6}
gradient_clipping=${7}
batch_size=${8}
early_stopping_patience=${9}
cross_attn_regularization=${10}
weight_decay=${11}
accumulation_steps=${12}
num_epochs=${13}
cross_attention_heads=${14}
advanced_masking=${15}
cross_attention_output_involevment=${16}
pretrained_model_dir=${17}
task=${18}
frozen_masking_layers=${19}
linear_probe=${20}

echo "Output directory: ${output_dir}"
echo "Masking strategy: ${masking_strategy}"
echo "Latent dim: ${latent_dim}"
echo "Num latents: ${num_latents}"
echo "Lr: ${lr}"
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
echo "Pretrained model directory: ${pretrained_model_dir}"
echo "Task: ${task}"
echo "Frozen masking layers: ${frozen_masking_layers}"
echo "Linear probe: ${linear_probe}"

RESUME="TRUE"


activate base 
srun torchrun --standalone --nproc_per_node=4 ../../main.py \
    --task ${task} \
    --masking_strategy ${masking_strategy} \
    --latent_dim ${latent_dim} \
    --num_latents ${num_latents} \
    --lr ${lr} \
    --gradient_clipping ${gradient_clipping} \
    --batch_size ${batch_size} \
    --patience ${early_stopping_patience} \
    --cross_attn_regularization ${cross_attn_regularization} \
    --weight_decay ${weight_decay} \
    --accumulation_steps ${accumulation_steps} \
    --max_epochs ${num_epochs} \
    --cross_attention_heads ${cross_attention_heads} \
    --advanced_masking ${advanced_masking} \
    --cross_attention_output_involevment ${cross_attention_output_involevment} \
    --train_data /h/am/EX_M/datasets/scRNA/merged/seed_32/scFoundation/train/bone_marrow_train_processed.h5ad \
    --val_data /h/am/EX_M/datasets/scRNA/merged/seed_32/scFoundation/validation/bone_marrow_val_processed.h5ad \
    --output_dir ${output_dir} \
    --pretrained_model_dir ${pretrained_model_dir} \
    --frozen_masking_layers ${frozen_masking_layers} \
    --original_model_dir /h/am/EX_M/scFoundation/pretrained_model/model.ckpt \
    --resume True \
    --linear_probe ${linear_probe} \
    &

trap " ${INPUT_PARENT_SCRIPT} \"${RESUME}\" " USR1

wait