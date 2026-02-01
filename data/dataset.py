from torch.utils.data import Dataset
import scanpy as sc
import torch

class PretrainDataset(Dataset):
    def __init__(self, data_path, transformer=None):
        self.data_path = data_path
        self.transform = transformer
        
        self.__data = sc.read_h5ad(self.data_path, backed='r')

    def __len__(self):
        return self.__data.shape[0] 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract data for the given index
        sample_id = self.__data.obs['ID'][idx]
        sample_sparse = self.__data[idx].X  # Sparse matrix

        # Convert sparse matrix to dense tensor (if feasible)
        sample = torch.tensor(sample_sparse.toarray(), dtype=torch.float).squeeze()

        if self.transform:
            sample = self.transform(sample)

        return sample_id, sample
    
    
class AnnotationDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_file_path (string): Path to the h5ad file.
            label_encoder (LabelEncoder)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.transform = transform
        
        
        self.adata = sc.read_h5ad(self.data_path, backed='r')
        

    def __len__(self):
        return self.adata.shape[0] 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract data for the given index
        sample_id = self.adata.obs['ID'][idx]
        sample_sparse = self.adata[idx].X  # Sparse matrix
        try:
            label = self.adata.obs['label'].iloc[idx]
        except:
            label = None
        

        # Convert sparse matrix to dense tensor (if feasible)
        sample = torch.tensor(sample_sparse.toarray(), dtype=torch.float).squeeze()
        
        if label is not None:
            label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)

        return sample_id, (sample, label)