import scanpy as sc
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
# sc.settings.verbosity = 3
# sc.logging.print_versions()

try:
    data_config = str(sys.argv[1])
except:
    print("Please provide the path to the data config file")
    sys.exit(1)
    
with open(data_config) as f:
    data_config = json.load(f)


merged_data = sc.read(data_config['path'] + '/bone_marrow.h5ad')

merged_data.layers["counts_temp"] = merged_data.X.copy() # preserve normalized counts
merged_data.X = merged_data.layers["counts"].copy()

sc.pp.filter_cells(merged_data, min_genes=200) # filter cells with less than 200 genes
sc.pp.filter_genes(merged_data, min_cells=3) # filter genes expressed in less than 3 cells
sc.pp.highly_variable_genes(merged_data, n_top_genes=data_config['top_genes'], flavor='seurat_v3')
highly_variable_genes = merged_data.var['highly_variable']

temp_index = merged_data.obs.index.copy()

merged_data.X = merged_data.layers["counts_temp"].copy()
del merged_data.layers["counts_temp"]

merged_data = merged_data[temp_index]

print("*"*10 + " Saving genes dictionary " + "*"*10)
merged_data.var.to_csv(data_config['path'] + "/genes.csv")

print("*"*10 + " Saving highly expressed genes " + "*"*10)
highly_variable_genes.to_csv(data_config['path'] + '/highly_variable_genes.csv')


print("*"*10 + " Merged data " + "*"*10)
print(merged_data.obs['donor'].value_counts())
print(merged_data.obs['timepoint'].value_counts())

to_split_df = pd.DataFrame({
    'donor': merged_data.obs['donor'],
    'celltype': merged_data.obs['celltype'],
    'timepoint': merged_data.obs['timepoint']
})
to_split_df.groupby(['donor'], observed=True)[['timepoint', 'celltype']].apply(lambda x: x.value_counts()).unstack().fillna(0)
random_state = data_config['seed']

# Define a new column that combines 'timepoint' and 'celltype' for stratification
to_split_df['strata'] = str(to_split_df['celltype']) + "_" + str(to_split_df['timepoint'])

# Create a unique list of donors
unique_donors = to_split_df['donor'].unique()

# Ensure each donor is grouped and a strata is selected safely
def safe_mode(series):
    if len(series.mode()) > 0:
        return series.mode()[0]
    else:
        return series.iloc[0]  # default to the first item if no mode


# Recompute the strata using the safe_mode function
donor_strata = to_split_df.groupby('donor', observed=False)['strata'].agg(safe_mode)

# Split donors into train, val, test
train_donors, temp_donors = train_test_split(unique_donors, test_size=0.2, random_state=random_state, stratify=donor_strata[unique_donors])

temp_strata = donor_strata[temp_donors]
val_donors, test_donors = train_test_split(temp_donors, test_size=0.5, random_state=random_state, stratify=temp_strata)

# Assign each donor to a set
to_split_df['set'] = to_split_df['donor'].apply(lambda x: 'train' if x in train_donors else ('val' if x in val_donors else 'test'));

print("*"*10 + " Train data " + "*"*10)
print(to_split_df[to_split_df['set'] == 'train']['celltype'].value_counts())
print(to_split_df[to_split_df['set'] == 'train']['timepoint'].value_counts())
print("*"*10 + " Test data " + "*"*10)
print(to_split_df[to_split_df['set'] == 'test']['celltype'].value_counts())
print(to_split_df[to_split_df['set'] == 'test']['timepoint'].value_counts())
print("*"*10 + " Val data " + "*"*10)
print(to_split_df[to_split_df['set'] == 'val']['celltype'].value_counts())
print(to_split_df[to_split_df['set'] == 'val']['timepoint'].value_counts())


merged_data.obs['set'] = to_split_df['set']

train_data = merged_data[merged_data.obs['set'] == 'train'].copy()
val_data = merged_data[merged_data.obs['set'] == 'val'].copy()
test_data = merged_data[merged_data.obs['set'] == 'test'].copy()


del merged_data, to_split_df
gc.collect();


# cell_name,cell_type,genes
def transform_data(df, max_genes=2000):
    transformed_data = []
    for row_idx in tqdm(range(df.shape[0])):
        name = df[row_idx].obs.index.values[0]
        ID = df[row_idx].obs['ID'].values[0]
        cell_type = df[row_idx].obs['celltype'].values[0]
        timepoint = df[row_idx].obs['timepoint'].values[0]
        
        non_zero_indices = set(np.nonzero(df[row_idx].X.toarray().squeeze())[0])
        genes = df[row_idx].var.index[[idx for idx in np.argsort(df[row_idx].X.toarray().squeeze())[::-1] if idx in non_zero_indices]][:max_genes].map(lambda x: x.upper())
        
        
        numbers = df[row_idx].X.toarray().squeeze()[[idx for idx in np.argsort(df[row_idx].X.toarray().squeeze())[::-1] if idx in non_zero_indices]][:max_genes].astype(str)
        genes_with_expression = ' '.join([f"{gene}#{number}" for gene, number in zip(genes, numbers)])
        
        transformed_data.append([ID, name, cell_type, timepoint, genes_with_expression])
        
        
    
    
    new_df = pd.DataFrame(transformed_data, columns=["ID", "cell_name", "cell_type", "timepoint", "genes"])
    return new_df

def process_batch(batch, df, max_genes):
    batch_result = []
    for row_idx in batch:
        name = df[row_idx].obs.index.values[0]
        ID = df[row_idx].obs['ID'].values[0]
        cell_type = df[row_idx].obs['celltype'].values[0]
        timepoint = df[row_idx].obs['timepoint'].values[0]

        non_zero_indices = set(np.nonzero(df[row_idx].X.toarray().squeeze())[0])
        genes = df[row_idx].var.index[[idx for idx in np.argsort(df[row_idx].X.toarray().squeeze())[::-1] if idx in non_zero_indices]][:max_genes].map(lambda x: x.upper())

        numbers = df[row_idx].X.toarray().squeeze()[[idx for idx in np.argsort(df[row_idx].X.toarray().squeeze())[::-1] if idx in non_zero_indices]][:max_genes].astype(str)
        genes_with_expression = ' '.join([f"{gene}#{number}" for gene, number in zip(genes, numbers)])
        batch_result.append([ID, name, cell_type, timepoint, genes_with_expression])
        
            
            
    return batch_result

def transform_data_multi_thread(df, max_genes=2000, num_threads=4):
    transformed_data = []
    batch_size = len(df) // num_threads

    batches = [range(i * batch_size, min((i + 1) * batch_size, len(df))) for i in range(num_threads)]
    

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_batch, batch, df, max_genes) for batch in batches]
        for future in tqdm(as_completed(futures), total=len(futures)):
            transformed_data.extend(future.result())
            
    
    
    new_df = pd.DataFrame(transformed_data, columns=["ID", "cell_name", "cell_type", "timepoint", "genes"])
    
    return new_df


print("*"*10 + " Saving raw val data " + "*"*10)
val_data.obs['ID'] = range(val_data.shape[0])
val_data.write(data_config['path'] + f'/seed_{random_state}/bone_marrow_val.h5ad')

if data_config['only_highly_variable']:
    val_data = val_data[:, highly_variable_genes]

print("*"*10 + " Transforming val data " + "*"*10)
transformed_val_data = transform_data_multi_thread(val_data, num_threads=40, max_genes=data_config['top_genes'])
del val_data
gc.collect();

transformed_val_data.to_csv(data_config['path'] + f'/seed_{random_state}/bone_marrow_val_with_expression{"_highly_variable" if data_config["only_highly_variable"] else ""}.csv', index=False)

del transformed_val_data
gc.collect();


print("*"*10 + " Saving raw test data " + "*"*10)
test_data.obs['ID'] = range(test_data.shape[0])
test_data.write(data_config['path'] + f'/seed_{random_state}/bone_marrow_test.h5ad')

if data_config['only_highly_variable']:
    test_data = test_data[:, highly_variable_genes]

print("*"*10 + " Transforming test data " + "*"*10)
transformed_test_data = transform_data_multi_thread(test_data, num_threads=40, max_genes=data_config['top_genes'])
del test_data
gc.collect();

transformed_test_data.to_csv(data_config['path'] + f'/seed_{random_state}/bone_marrow_test_with_expression{"_highly_variable" if data_config["only_highly_variable"] else ""}.csv', index=False)

del transformed_test_data
gc.collect();


print("*"*10 + " Saving raw train data " + "*"*10)
train_data.obs['ID'] = range(train_data.shape[0])
train_data.write(data_config['path'] + f'/seed_{random_state}/bone_marrow_train.h5ad')

if data_config['only_highly_variable']:
    train_data = train_data[:, highly_variable_genes]

print("*"*10 + " Transforming train data " + "*"*10)
transformed_train_data = transform_data_multi_thread(train_data, num_threads=40, max_genes=data_config['top_genes'])
del train_data
gc.collect();

transformed_train_data.to_csv(data_config['path'] + f'/seed_{random_state}/bone_marrow_train_with_expression{"_highly_variable" if data_config["only_highly_variable"] else ""}.csv', index=False)

del transformed_train_data
gc.collect();
print("done!")