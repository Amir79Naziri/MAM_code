## Modeified code from https://github.com/biomap-research/scFoundation

import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from einops import rearrange, repeat
from functools import partial
from contextlib import contextmanager


def exists(val):
    return val is not None

class AutoDiscretizationEmbedding2(nn.Module):
    def __init__(self, dim, max_seq_len, bin_num, bin_alpha, mask_token_id = None, pad_token_id = None):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha
        
        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)
        
        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)
        
        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        # print('self.bin_num_idx',self.bin_num_idx, self.bin_num_idx.shape)

        self.tensor0 = torch.tensor(0, dtype=torch.long)

    def forward(self, x, output_weight=0):
        x_mask_idx = (x==self.mask_token_id).nonzero()
        x_pad_idx = (x==self.pad_token_id).nonzero()
        # print("x_mask",x_mask_idx.shape,x_mask_idx)
        
        x = self.mlp(x) # [B,N,1] -> [B,N,H]
        x = self.LeakyReLU(x) # [B,N,H]
        x_crosslayer = self.mlp2(x) # [B,N,H]
        x = self.bin_alpha * x + x_crosslayer # [B,N,H]
        weight = self.Softmax(x) # [B, N, H]
        # print('weight', weight.shape, weight, torch.sum(weight, 2))
        
        bin_num_idx = self.bin_num_idx.to(x.device) # [H,]
        # print('bin_num_idx', bin_num_idx.shape)
        
        token_emb = self.emb(bin_num_idx) # [H, D]
        # print('token_emb', token_emb.shape)
        x = torch.matmul(weight, token_emb) #[B, N, D]
    
        # print("x_emb",x.shape,x)
        
        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)

        mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)
        # print(mask_token_emb.dtype)
        # print("x", x.dtype)
        x[x_mask_idx[:,0],x_mask_idx[:,1],:] = mask_token_emb.repeat(x_mask_idx.shape[0],1)
        # print("x_emb",x.shape,x)

        pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        x[x_pad_idx[:,0],x_pad_idx[:,1],:] = pad_token_emb.repeat(x_pad_idx.shape[0],1)
    
        if output_weight:
            return x,weight
        return x

class RandomPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)


class MaeAutobin(nn.Module):
    def __init__(
            self,
            *,
            num_tokens,  # num of tokens
            max_seq_len,  # max length of sequence
            embed_dim,  # encoder dim of tokens
            decoder_embed_dim,
            tie_embed=False,
            bin_alpha = 1.0,
            bin_num = 10,
            latent_dim=512,
            num_latents=256,
            pad_token_id = None,
            mask_token_id = None,
            advanced_masking=False,
            masking_ratio=0.3 * 0.07,
            max_tokens=6000,
            cross_attention_heads=8,
            masking_strategy='cross-attention',
            cross_attention_output_involevment='both',
    ):
        super(MaeAutobin, self).__init__()

        self.max_seq_len = max_seq_len
        self.num_tokens = num_tokens
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.advanced_masking = advanced_masking
        self.masking_ratio = masking_ratio
        self.max_tokens = max_tokens
        self.masking_strategy = masking_strategy
        self.cross_attention_output_involevment = cross_attention_output_involevment
        if self.masking_strategy == "cross-attention":
            self.latent_dim = latent_dim  # Dimension of latent vectors
            self.num_latents = num_latents  # Number of latent vectors

            # Initialize latent vectors
            self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))

            # Cross-attention parameters
            self.cross_attention = nn.MultiheadAttention(embed_dim=self.latent_dim, num_heads=cross_attention_heads)

            # Projection layers to match dimensions
            self.input_proj = nn.Linear(1, self.latent_dim)
            if self.cross_attention_output_involevment == 'separate':
                self.output_proj_enc = nn.Linear(self.latent_dim, embed_dim)
                self.output_proj_dec = nn.Linear(self.latent_dim, embed_dim)
            else:
                self.output_proj = nn.Linear(self.latent_dim, embed_dim)

        # encoder
        self.token_emb = AutoDiscretizationEmbedding2(embed_dim, max_seq_len, bin_num=bin_num, bin_alpha=bin_alpha, pad_token_id=self.pad_token_id, mask_token_id=self.mask_token_id)
        self.pos_emb = nn.Embedding(max_seq_len+1, embed_dim)  #RandomPositionalEmbedding(embed_dim, max_seq_len)

        # ## DEBUG
        self.encoder = None

        ##### decoder
        self.decoder = None
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.to_final = nn.Linear(decoder_embed_dim, 1)
        
    def initialize_new_parameters(self):
        if self.masking_strategy == "cross-attention":
            if self.cross_attention_output_involevment == 'separate':
                vars = [self.latents, self.cross_attention, self.input_proj, self.output_proj_enc, self.output_proj_dec]
            else:
                vars = [self.latents, self.cross_attention, self.input_proj, self.output_proj]
            for m in vars:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            # Initialize latents
            nn.init.normal_(self.latents, mean=0.0, std=0.02)
        
    def generate_attention_mask(self, attn_weights, masking_ratio):
        # attn_weights: [batch_size, num_heads, num_latents, seq_len]
        batch_size, num_heads, num_latents, seq_len = attn_weights.size()
        device = attn_weights.device

        # Sum over heads to aggregate attentions
        attn_weights_summed = attn_weights.sum(dim=1)  # [batch_size, num_latents, seq_len]

        # Randomly select a subset of latents R (indices)
        num_selected_latents = int(num_latents * masking_ratio)
        R = torch.randperm(num_latents, device=device)[:num_selected_latents]

        # Sum attention maps over selected latents
        summed_attentions = attn_weights_summed[:, R, :].sum(dim=1)  # [batch_size, seq_len]

        # Select top-k tokens to mask
        k = int(seq_len * masking_ratio)
        _, topk_indices = torch.topk(summed_attentions, k=k, dim=-1)

        # Create mask
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        attention_mask.scatter_(1, topk_indices, False)  # False indicates positions to be masked

        return attention_mask  # [batch_size, seq_len]
    
    
    def generate_multinomial_attention_mask(self, attn_weights, masking_ratio):
        batch_size, num_heads, num_latents, seq_len = attn_weights.size()
        device = attn_weights.device

        # Sum over heads to aggregate attentions
        attn_weights_summed = attn_weights.sum(dim=1)  # [batch_size, num_latents, seq_len]

        # Randomly select a subset of latents R (indices)
        num_selected_latents = int(num_latents * masking_ratio)
        R = torch.randperm(num_latents, device=device)[:num_selected_latents]

        # Sum attention maps over selected latents
        summed_attentions = attn_weights_summed[:, R, :].sum(dim=1)  # [batch_size, seq_len]

        # Normalize summed attentions to get probabilities
        summed_attentions = summed_attentions + 1e-8  # Prevent division by zero
        probs = summed_attentions / summed_attentions.sum(dim=1, keepdim=True)

        # Sample tokens to mask
        k = int(seq_len * masking_ratio)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            masked_indices = torch.multinomial(probs[i], num_samples=k, replacement=False)
            attention_mask[i, masked_indices] = False  # False indicates positions to be masked

        return attention_mask  # [batch_size, seq_len]
    
    def random_masking(self, data, mask_ratio):
        batch_size, seq_len = data.shape
        device = data.device

        # Initialize mask tensor with all True (no masking)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # Mask for zero and non-zero tokens separately
        for i in range(batch_size):
            # Get indices for zero and non-zero tokens
            zero_indices = (data[i] == 0).nonzero(as_tuple=False).squeeze(1)
            non_zero_indices = (data[i] != 0).nonzero(as_tuple=False).squeeze(1)

            # Calculate number of tokens to mask for each category
            num_zero_to_mask = int(len(zero_indices) * mask_ratio)
            num_non_zero_to_mask = int(len(non_zero_indices) * mask_ratio)

            # Randomly select indices to mask within each category
            if num_zero_to_mask > 0:
                zero_mask_indices = zero_indices[torch.randperm(len(zero_indices))[:num_zero_to_mask]]
                mask[i, zero_mask_indices] = False  # False indicates positions to be masked

            if num_non_zero_to_mask > 0:
                non_zero_mask_indices = non_zero_indices[torch.randperm(len(non_zero_indices))[:num_non_zero_to_mask]]
                mask[i, non_zero_mask_indices] = False  # False indicates positions to be masked

        return mask  # [batch_size, seq_len]
    
    def prepare_data(self, data, only_encoder=False, device=None):
        if not only_encoder:
            decoder_data = data.clone().detach()
            decoder_data_padding = torch.full_like(data, False, dtype=torch.bool).to(device)

        encoder_data = data  
        encoder_data_padding = torch.full_like(encoder_data, False, dtype=torch.bool).to(device)

        data_gene_ids = torch.arange(data.shape[1], device=device).repeat(data.shape[0], 1)  # Shape: [batch_size, 19264]
        encoder_position_gene_ids = data_gene_ids
        if not only_encoder:
            decoder_position_gene_ids = data_gene_ids

        encoder_position_gene_ids[encoder_data_padding] = self.max_seq_len 
        if not only_encoder:
            decoder_position_gene_ids[decoder_data_padding] = self.max_seq_len 
        
        if only_encoder:
            return encoder_data, encoder_data_padding, encoder_position_gene_ids
        else:
            return encoder_data, encoder_data_padding, encoder_position_gene_ids, decoder_data, decoder_data_padding, decoder_position_gene_ids
        
    def gatherData(self, data, labels, pad_token_id):
        value_nums = labels.sum(1)
        max_num = max(value_nums)


        fake_data = torch.full((data.shape[0], max_num), pad_token_id,
                            device=data.device)
        data = torch.hstack([data, fake_data])

        fake_label = torch.full((labels.shape[0], max_num), 1,
                                device=labels.device)
        none_labels = ~labels
        labels = labels.float()
        labels[none_labels] = torch.tensor(-float('Inf'), device=labels.device)

        tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
        labels += tmp_data

        labels = torch.hstack([labels, fake_label])

        fake_label_gene_idx = labels.topk(max_num).indices

        new_data = torch.gather(data, 1, fake_label_gene_idx)

        padding_labels = (new_data == pad_token_id)

        return new_data, padding_labels

    def forward(self, x, output_maskings=True, **kwargs):
        
        
        b, _ = x.shape  # b = batch_size
        encoder_data, encoder_data_padding, encoder_position_gene_ids, \
        decoder_data, decoder_data_padding, decoder_position_gene_ids = self.prepare_data(x, device=x.device)
        
        # *********************************************** Finding Masks ***********************************************
        if self.masking_strategy == "cross-attention":
            # Project encoder data to latent dimension per token
            encoder_data_proj = encoder_data.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
                
            x_proj = self.input_proj(encoder_data_proj)  # Shape: [batch_size, seq_len, latent_dim]

            # Prepare latents
            latents = self.latents.unsqueeze(0).expand(b, -1, -1)  # Shape: [batch_size, num_latents, latent_dim]

            # Cross-Attention: Latents attend to inputs
            # Transpose dimensions to match MultiheadAttention expectations
            # MultiheadAttention expects [seq_len, batch_size, embed_dim]

            x_proj = x_proj.permute(1, 0, 2)  # [seq_len, batch_size, latent_dim]
            latents = latents.permute(1, 0, 2)  # [num_latents, batch_size, latent_dim]

            
            
            # Compute cross-attention
            cross_attn_output, cross_attn_weights = self.cross_attention(
                query=latents,
                key=x_proj,
                value=x_proj,
                key_padding_mask=encoder_data_padding,
                need_weights=True,
                average_attn_weights=False
            ) # cross_attn_weights: [batch_size, num_heads, num_latents, seq_len]
            assert not torch.isnan(cross_attn_output).any(), "cross_attention contain NaN"
            
            # Transpose back
            cross_attn_output = cross_attn_output.permute(1, 0, 2)  # Shape: [batch_size, num_latents, latent_dim]
            cross_attn_output_pooled = cross_attn_output.mean(dim=1) # Shape: [batch_size, latent_dim]
            
            if self.cross_attention_output_involevment == 'separate':
                cross_attn_output_proj_enc = self.output_proj_enc(cross_attn_output_pooled).unsqueeze(1)  # Shape: [batch_size, 1, embed_dim]
                cross_attn_output_proj_dec = self.output_proj_dec(cross_attn_output_pooled).unsqueeze(1)  # Shape: [batch_size, 1, embed_dim]
            else:
                cross_attn_output_proj = self.output_proj(cross_attn_output_pooled).unsqueeze(1)  # Shape: [batch_size, 1, embed_dim]
            # Generate attention mask
            mask = self.generate_multinomial_attention_mask(cross_attn_weights, masking_ratio=self.masking_ratio)  # [batch_size, seq_len]
        elif self.masking_strategy == "random":
            mask = self.random_masking(encoder_data, mask_ratio=self.masking_ratio)
        
        # *********************************************** Encoder ***********************************************
        # Mask encoder data to remove masks and zeros
        encoder_data_masked = encoder_data.clone()
        
        # print("non-mask ratio", encoder_data_masked[mask].shape)
        # print("mask ratio", encoder_data_masked[~mask].shape)
        
        # print("non-mask ratio (non-zero)", (encoder_data_masked[mask] != 0).sum())
        # print("mask ratio (non-zero)", (encoder_data_masked[~mask] != 0).sum())
        
        # print("non-mask ratio (zero)", (encoder_data_masked[mask] == 0).sum())
        # print("mask ratio (zero)", (encoder_data_masked[~mask] == 0).sum())
        
        # Only Mask if there are more non-masked tokens than zero tokens
        for i in range(b):
            if len(encoder_data_masked[i, mask[i]]) > (encoder_data_masked[i, mask[i]] == 0).sum():
                encoder_data_masked[i, ~mask[i]] = self.mask_token_id
            else:
                mask[i, :] = 1
        
        encoder_data_labels = (encoder_data_masked > 0) & (encoder_data_masked != self.mask_token_id)
        
        # Calculate the number of unmasked tokens per sequence
        num_unmasked = encoder_data_labels.sum(dim=1)  # Shape: [batch_size]

        # Define the maximum allowed tokens
        max_allowed_tokens = self.max_tokens

        # Determine excess tokens
        excess_tokens = num_unmasked - max_allowed_tokens
        excess_tokens = torch.clamp(excess_tokens, min=0)  # Ensure non-negative

        # Iterate over the batch to apply random masking where necessary
        for i in range(b):
            if excess_tokens[i] > 0:
                # Get indices of unmasked tokens
                unmasked_indices = encoder_data_labels[i].nonzero(as_tuple=False).squeeze(1)
                
                # Randomly select indices to mask
                random_mask_indices = unmasked_indices[torch.randperm(unmasked_indices.size(0))[:excess_tokens[i]]]
                
                # Apply masking
                encoder_data_masked[i, random_mask_indices] = self.mask_token_id
                mask[i, random_mask_indices] = 0
                
                # Update labels
                encoder_data_labels[i, random_mask_indices] = False
            
        
        # Remove zeros and masked tokens after masking (if necessary)
        encoder_data_adjusted, encoder_data_padding = self.gatherData(encoder_data_masked, encoder_data_labels,
                                                        self.pad_token_id)

        encoder_position_gene_ids, _ = self.gatherData(encoder_position_gene_ids, encoder_data_labels,
                                                self.pad_token_id)

        encoder_position_gene_ids[encoder_data_padding] = self.max_seq_len
        

        # Now proceed with token and positional embeddings
        x = self.token_emb(torch.unsqueeze(encoder_data_adjusted, 2), output_weight=0)
        position_emb = self.pos_emb(encoder_position_gene_ids)
        if self.masking_strategy == "cross-attention":
            if self.cross_attention_output_involevment == 'separate':
                x_embedded = x + position_emb + cross_attn_output_proj_enc
            elif self.cross_attention_output_involevment == 'both' or self.cross_attention_output_involevment == 'encoder':
                x_embedded = x + position_emb + cross_attn_output_proj
            else:
                x_embedded = x + position_emb
        elif self.masking_strategy == "random":
            x_embedded = x + position_emb

        # Pass through encoder
        x_encoded = self.encoder(x_embedded, encoder_data_padding)
        
        # *********************************************** Decoder ***********************************************

        # Mask decoder data
        if self.advanced_masking:
            # Apply mask to decoder_data before embeddings with multiple scenarios
            mask_indices = (~mask).nonzero(as_tuple=False)
            decoder_data_masked = decoder_data.clone()

            # # Define replacement probabilities
            replace_prob = torch.rand(mask_indices.size(0), device=decoder_data_masked.device, dtype=decoder_data.dtype)
            mask_replace = replace_prob < 0.8
            random_replace = (replace_prob >= 0.8) & (replace_prob < 0.9)

            # # 80% Replace with mask_token_id
            decoder_data_masked[mask_indices[mask_replace, 0], mask_indices[mask_replace, 1]] = self.mask_token_id

            # # 10% Replace with random token_ids
            num_random = random_replace.sum()
            if num_random > 0:
                random_tokens = torch.rand(num_random, device=encoder_data.device, dtype=decoder_data.dtype) * decoder_data.max()
                decoder_data_masked[mask_indices[random_replace, 0], mask_indices[random_replace, 1]] = random_tokens

            # 10% Keep original tokens (no action needed)
        else:
            decoder_data_masked = decoder_data.clone()
            decoder_data_masked[~mask] = self.mask_token_id
            
        # decoder_data_masked = decoder_data_masked.float()
        
        d_embedded = self.token_emb(torch.unsqueeze(decoder_data_masked, 2))
        position_emb = self.pos_emb(decoder_position_gene_ids)

        batch_idx, gen_idx = (encoder_data_labels == True).nonzero(as_tuple=True) 

        d_embedded[batch_idx, gen_idx] = x_encoded[~encoder_data_padding].to(d_embedded.dtype)

        if self.masking_strategy == "cross-attention":
            if self.cross_attention_output_involevment == 'separate':
                d_embedded += position_emb + cross_attn_output_proj_dec
            elif self.cross_attention_output_involevment == 'both' or self.cross_attention_output_involevment == 'decoder':
                d_embedded += position_emb + cross_attn_output_proj
            else:
                d_embedded += position_emb
        elif self.masking_strategy == "random":
            d_embedded += position_emb

        d_embedded2 = self.decoder_embed(d_embedded)

        # Pass through decoder
        out = self.decoder(d_embedded2, padding_mask=decoder_data_padding)

        out_normed = self.norm(out)

        # print("x1",x.shape) 
        final_out = self.to_final(out_normed)
        
        if self.masking_strategy == "cross-attention":
            return final_out.squeeze(2), mask, cross_attn_output
        elif self.masking_strategy == "random":
            return final_out.squeeze(2), mask
         
        
class CellAnnoClassifierCA(nn.Module):

    def __init__(self, n_class, n_genes, model, model_config, frozen_masking_layers=False, linear_probe=False):
        super().__init__()
        self.n_class = n_class
        self.n_genes = n_genes
        
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        
        
        
        # check if separate output projection is used
        if model.cross_attention_output_involevment == 'separate':
            self.output_proj = model.output_proj_enc
        elif model.cross_attention_output_involevment == 'encoder' or model.cross_attention_output_involevment == 'both':
            self.output_proj = model.output_proj
            
        self.cross_attention_output_involevment = (model.cross_attention_output_involevment == 'separate') or \
                                                    (model.cross_attention_output_involevment == 'both') or \
                                                    (model.cross_attention_output_involevment == 'encoder')
                                                    
        if self.cross_attention_output_involevment:
            self.latents = model.latents
            self.cross_attention = model.cross_attention
            self.input_proj = model.input_proj
            
            if frozen_masking_layers:
                self.latents.requires_grad = False
                
                for _,p in self.cross_attention.named_parameters():
                    p.requires_grad = False
                for _,p in self.input_proj.named_parameters():
                    p.requires_grad = False
                for _,p in self.output_proj.named_parameters():
                    p.requires_grad = False
            
            else:
                self.latents.requires_grad = True
                
                for _,p in self.cross_attention.named_parameters():
                    p.requires_grad = True
                for _,p in self.input_proj.named_parameters():
                    p.requires_grad = True
                for _,p in self.output_proj.named_parameters():
                    p.requires_grad = True
                
            
        if linear_probe:
            for _,p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _,p in self.pos_emb.named_parameters():
                p.requires_grad = False

            for _, p in self.encoder.named_parameters():
                p.requires_grad = False
            for _, p in self.encoder.transformer_encoder[-2].named_parameters():
                p.requires_grad = False
        else:         
            for _,p in self.token_emb.named_parameters():
                p.requires_grad = True
            for _,p in self.pos_emb.named_parameters():
                p.requires_grad = True
            
            for _, p in self.encoder.named_parameters():
                p.requires_grad = False
            for _, p in self.encoder.transformer_encoder[-2].named_parameters():
                p.requires_grad = True


        self.fc1 = nn.Sequential(
        nn.Linear(model_config['encoder']['hidden_dim'], 256),
        nn.ReLU(),
        nn.Linear(256, self.n_class)  # ['n_class']
        ) 
        self.norm = torch.nn.BatchNorm1d(model_config['encoder']['hidden_dim'], affine=False, eps=1e-6)
        self.model_config = model_config        
        
    def gatherData(self, data, labels, pad_token_id):
        value_nums = labels.sum(1)
        max_num = max(value_nums)


        fake_data = torch.full((data.shape[0], max_num), pad_token_id,
                            device=data.device)
        data = torch.hstack([data, fake_data])

        fake_label = torch.full((labels.shape[0], max_num), 1,
                                device=labels.device)
        none_labels = ~labels
        labels = labels.float()
        labels[none_labels] = torch.tensor(-float('Inf'), device=labels.device)

        tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
        labels += tmp_data

        labels = torch.hstack([labels, fake_label])

        fake_label_gene_idx = labels.topk(max_num).indices

        new_data = torch.gather(data, 1, fake_label_gene_idx)

        padding_labels = (new_data == pad_token_id)

        return new_data, padding_labels
        
    def forward(self, x, return_embedding=False, **kwargs):
        
        b, _ = x.shape
        
        
        if self.cross_attention_output_involevment:
            x_proj = self.input_proj(x.unsqueeze(-1))  # Shape: [batch_size, seq_len, latent_dim]
            
            latents = self.latents.unsqueeze(0).expand(b, -1, -1)  # Shape: [batch_size, num_latents, latent_dim]

            x_proj = x_proj.permute(1, 0, 2)  # [seq_len, batch_size, latent_dim]
            latents = latents.permute(1, 0, 2)  # [num_latents, batch_size, latent_dim]

            
            
            # Compute cross-attention
            cross_attn_output, _ = self.cross_attention(
                query=latents,
                key=x_proj,
                value=x_proj,
                key_padding_mask=None,
                need_weights=False,
                average_attn_weights=False
            ) # cross_attn_weights: [batch_size, num_heads, num_latents, seq_len]
            
            # print("cross_attn_output", cross_attn_output)
            assert not torch.isnan(cross_attn_output).any(), "cross_attention contain NaN"
        
            # Transpose back
            cross_attn_output = cross_attn_output.permute(1, 0, 2)  # Shape: [batch_size, num_latents, latent_dim]
            cross_attn_output_pooled = cross_attn_output.mean(dim=1) # Shape: [batch_size, latent_dim]
            cross_attn_output_proj = self.output_proj(cross_attn_output_pooled).unsqueeze(1)
        
        
        value_labels = x > 0
        x, x_padding = self.gatherData(x, value_labels, self.model_config['pad_token_id'])
        data_gene_ids = torch.arange(self.n_genes, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = self.gatherData(data_gene_ids, value_labels,
                                        self.model_config['pad_token_id'])
        
        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
        position_emb = self.pos_emb(position_gene_ids)
        
        if self.cross_attention_output_involevment:
            x += position_emb + cross_attn_output_proj
        else:
            x += position_emb

        try:
            logits = self.encoder(x,x_padding)
        except:
            print("x", x.shape)
            print("x_padding", x_padding.shape)

        # mlp
        logits, _ = torch.max(logits, dim=1)  # b,dim
        
        if return_embedding:
            embeddings = logits.clone()

        if logits.size(0) > 1:  # Only apply BatchNorm if batch size > 1
            logits = self.norm(logits)
            
        logits = self.fc1(logits)

        if self.cross_attention_output_involevment:
            if return_embedding:
                return logits, embeddings, cross_attn_output
            return logits, cross_attn_output
        
        if return_embedding:
            return logits, embeddings
        return logits


class CellAnnoClassifierR(nn.Module):

    def __init__(self, n_class, n_genes, model, model_config, linear_probe=False):
        super().__init__()
        self.n_class = n_class
        self.n_genes = n_genes
        
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        
        if linear_probe:
            for _,p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _,p in self.pos_emb.named_parameters():
                p.requires_grad = False

            for _, p in self.encoder.named_parameters():
                p.requires_grad = False
            for _, p in self.encoder.transformer_encoder[-2].named_parameters():
                p.requires_grad = False
        else:
            for _,p in self.token_emb.named_parameters():
                p.requires_grad = True
            for _,p in self.pos_emb.named_parameters():
                p.requires_grad = True
            
            for _, p in self.encoder.named_parameters():
                p.requires_grad = False
            for _, p in self.encoder.transformer_encoder[-2].named_parameters():
                p.requires_grad = True


        self.fc1 = nn.Sequential(
        nn.Linear(model_config['encoder']['hidden_dim'], 256),
        nn.ReLU(),
        nn.Linear(256, self.n_class)  # ['n_class']
        ) 
        self.norm = torch.nn.BatchNorm1d(model_config['encoder']['hidden_dim'], affine=False, eps=1e-6)
        self.model_config = model_config
        
    def gatherData(self, data, labels, pad_token_id):
        value_nums = labels.sum(1)
        max_num = max(value_nums)


        fake_data = torch.full((data.shape[0], max_num), pad_token_id,
                            device=data.device)
        data = torch.hstack([data, fake_data])

        fake_label = torch.full((labels.shape[0], max_num), 1,
                                device=labels.device)
        none_labels = ~labels
        labels = labels.float()
        labels[none_labels] = torch.tensor(-float('Inf'), device=labels.device)

        tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
        labels += tmp_data

        labels = torch.hstack([labels, fake_label])

        fake_label_gene_idx = labels.topk(max_num).indices

        new_data = torch.gather(data, 1, fake_label_gene_idx)

        padding_labels = (new_data == pad_token_id)

        return new_data, padding_labels
                    
        
    def forward(self, x, return_embedding=False, **kwargs):
        value_labels = x > 0
        x, x_padding = self.gatherData(x, value_labels, self.model_config['pad_token_id'])
        data_gene_ids = torch.arange(self.n_genes, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = self.gatherData(data_gene_ids, value_labels,
                                        self.model_config['pad_token_id'])
        
        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb

        logits = self.encoder(x,x_padding)

        # mlp
        logits, _ = torch.max(logits, dim=1)  # b,dim
        
        if return_embedding:
            embeddings = logits.clone()

        if logits.size(0) > 1:  # Only apply BatchNorm if batch size > 1
            logits = self.norm(logits)
            
        logits = self.fc1(logits)

        if return_embedding:
            return logits, embeddings
        return logits