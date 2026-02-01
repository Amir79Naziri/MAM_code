# this file is used to prepare the data for scFoundation model 
# Modeified code from https://github.com/biomap-research/scFoundation
from scRNA_workflow import *
import os
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

# dataset_name = 'bone_marrow'
# dataset_subname = "merged"

dataset_name = 'retina'
dataset_subname = "retina"


sc.settings.figdir='./figures_new/'

train_adata = sc.read_h5ad(f'/local/home/am/EX_M/datasets/scRNA/{dataset_subname}/seed_32/{dataset_name}_train.h5ad')
test_adata = sc.read_h5ad(f'/local/home/am/EX_M/datasets/scRNA/{dataset_subname}/seed_32/{dataset_name}_test.h5ad')
val_adata = sc.read_h5ad(f'/local/home/am/EX_M/datasets/scRNA/{dataset_subname}/seed_32/{dataset_name}_val.h5ad')


def preprocess(adata):
    # comment below if data index is gene symbols, otherwise you should mention the column name that contains gene symbols
    adata.var.set_index('feature_name', inplace=True)
    
    X_df= pd.DataFrame(sparse.csr_matrix.toarray(adata.X),index=adata.obs.index.tolist(),columns=adata.var.index.tolist())
    gene_list_df = pd.read_csv('./OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])
    X_df, _, _ = main_gene_selection(X_df, gene_list)
    adata_uni = sc.AnnData(X_df)
    adata_uni.X = sparse.csr_matrix(adata_uni.X).astype(dtype='float32')
    
    adata_uni.obs = adata.obs
    adata_uni.uns = adata.uns
    
    return adata_uni

os.makedirs(f'/local/home/am/EX_M/datasets/scRNA/{dataset_subname}/seed_32/scFoundation/', exist_ok=True)
os.makedirs(f'/local/home/am/EX_M/datasets/scRNA/{dataset_subname}/seed_32/scFoundation/train/', exist_ok=True)
os.makedirs(f'/local/home/am/EX_M/datasets/scRNA/{dataset_subname}/seed_32/scFoundation/test/', exist_ok=True)
os.makedirs(f'/local/home/am/EX_M/datasets/scRNA/{dataset_subname}/seed_32/scFoundation/validation/', exist_ok=True)

print("Train data preprocess")
train_adata_processed = preprocess(train_adata)

le = LabelEncoder().fit(train_adata_processed.obs['celltype'])
train_adata_processed.obs['label'] = le.transform(train_adata_processed.obs['celltype'])

train_adata_processed.write_h5ad(f'/local/home/am/EX_M/datasets/scRNA/{dataset_subname}/seed_32/scFoundation/train/{dataset_name}_train_processed.h5ad')

print("Val data preprocess")
val_adata_processed = preprocess(val_adata)

val_adata_processed.obs['label'] = le.transform(val_adata_processed.obs['celltype'])

val_adata_processed.write_h5ad(f'/local/home/am/EX_M/datasets/scRNA/{dataset_subname}/seed_32/scFoundation/validation/{dataset_name}_val_processed.h5ad')

print("Test data preprocess")
test_adata_processed = preprocess(test_adata)

test_adata_processed.obs['label'] = le.transform(test_adata_processed.obs['celltype'])

test_adata_processed.write_h5ad(f'/local/home/am/EX_M/datasets/scRNA/{dataset_subname}/seed_32/scFoundation/test/{dataset_name}_test_processed.h5ad')