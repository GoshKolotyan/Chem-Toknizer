import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging
from time import time

logger = logging.getLogger("ChemDataLoader")

class ChemDataset(Dataset):
    """
    Custom dataset for chemical formula data that handles charge imbalanced samples
    by filtering them out during initialization.
    """
    def __init__(self, csv_path, tokenizer, shuffle=True):
        """
        Initialize the dataset.
        
        Args:
            csv_path (str): Path to CSV file containing chemical data
            tokenizer (ChemTokenizer): Initialized tokenizer to process formulas
            max_samples (int, optional): Maximum number of samples to use (for debugging)
            shuffle (bool): Whether to shuffle the data before processing
        """
        self.logger = logging.getLogger("ChemDataset")
        self.logger.info(f"Initializing ChemDataset from {csv_path}")
        
        # Load data
        self.df = pd.read_csv(csv_path)
        self.logger.info(f"Loaded {len(self.df)} total samples from CSV")
        

        # Initialize tokenizer
        self.tokenizer = tokenizer
        
        # Pre-process all samples to filter out charge-imbalanced formulas
        self.valid_samples = []
        self.valid_indices = []
        skipped_count = 0
        
        start_time = time()
        self.logger.info("Pre-processing samples to filter out charge-imbalanced formulas")
        
        for i, row in self.df.iterrows():
            tensor_data = self.tokenizer.encode(row)
            if tensor_data is not None:
                X, y = tensor_data
                self.valid_samples.append((X, y))
                self.valid_indices.append(i)
            else:
                skipped_count += 1
                
        end_time = time()
        self.logger.info(f"Processed all samples in {end_time - start_time:.2f} seconds")
        self.logger.info(f"Kept {len(self.valid_samples)} valid samples, skipped {skipped_count} samples with charge imbalance")
        
        # Create a mapping from valid index to original index for reference
        self.index_mapping = dict(enumerate(self.valid_indices))
    
    def __len__(self):
        """Return the number of valid samples in the dataset."""
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        """Return the tensor data for a valid sample by index."""
        return self.valid_samples[idx]
    
    def get_original_row(self, idx):
        """Get the original DataFrame row for a given valid index."""
        original_idx = self.valid_indices[idx]
        return self.df.iloc[original_idx]

class ChemDataLoader:
    """
    Wrapper for PyTorch DataLoader with additional functionality for chemical data.
    """
    def __init__(self, csv_path, tokenizer, batch_size=32, shuffle=True, 
                 num_workers=0, max_samples=None, pin_memory=False):
        """
        Initialize the data loader.
        
        Args:
            csv_path (str): Path to CSV file containing chemical data
            tokenizer (ChemTokenizer): Initialized tokenizer to process formulas
            batch_size (int): Batch size for training
            shuffle (bool): Whether to shuffle the data
            num_workers (int): Number of worker processes for data loading
            max_samples (int, optional): Maximum number of samples to use (for debugging)
            pin_memory (bool): Whether to pin memory for faster GPU transfer
        """
        self.logger = logging.getLogger("ChemDataLoader")
        self.logger.info(f"Initializing ChemDataLoader with batch_size={batch_size}, shuffle={shuffle}")
        
        # Create dataset
        self.dataset = ChemDataset(csv_path, tokenizer, shuffle)
        
        # Create PyTorch DataLoader with custom collate function
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory
        )
    
    def collate_fn(self, batch):
        """
        Custom collate function to handle batches with different tensor shapes.
        
        Args:
            batch: List of (X, y) tuples
            
        Returns:
            Tuple of (X_batch, y_batch) where:
                X_batch is a padded tensor of shape (batch_size, max_seq_len)
                y_batch is a tensor of shape (batch_size)
        """
        # Separate X and y
        X_list = [item[0] for item in batch]
        y_list = [item[1] for item in batch]
        
        # Get max sequence length in this batch
        max_len = max(x.shape[0] for x in X_list)
        
        # Pad sequences to the same length
        X_padded = []
        for x in X_list:
            if x.shape[0] < max_len:
                # Pad with zeros
                padding = torch.zeros(max_len - x.shape[0], dtype=x.dtype)
                x_padded = torch.cat([x, padding])
            else:
                x_padded = x
            X_padded.append(x_padded)
        
        # Stack into batch tensors
        X_batch = torch.stack(X_padded)
        y_batch = torch.tensor(y_list)
        
        return X_batch, y_batch
    
    def __iter__(self):
        """Return an iterator over the dataloader."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Return the number of batches in the dataloader."""
        return len(self.dataloader)

# Example usage
if __name__ == "__main__":
    from tokenizer import ChemTokenizer  # Import your tokenizer
    import numpy as np
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize tokenizer
    tokenizer = ChemTokenizer()
    
    # Create data loader with a small subset for testing
    data_loader = ChemDataLoader(
        csv_path="Data/Full_Data_ranamed_columns.csv",
        tokenizer=tokenizer,
        batch_size=16,
        shuffle=False,
    )
    
    # Iterate through batches
    for batch_idx, (X_batch, y_batch) in enumerate(data_loader):
        print(f"\nBatch {batch_idx+1}/{len(data_loader)}")
        print(f"X shape: {X_batch.shape}, y shape: {y_batch.shape}")
        
        # Print each sample in the first batch with its feature vector and bandgap
        if batch_idx == 0:
            for i in range(min(10, len(X_batch))):
                # Calculate the actual dataset index for this item in the batch
                dataset_idx = batch_idx * data_loader.dataloader.batch_size + i
                # Get the original index
                original_idx = data_loader.dataset.valid_indices[dataset_idx]
                # Get the original row
                original_row = data_loader.dataset.df.iloc[original_idx]
                # print(data_loader.dataset.valid_indices)
                
                # Get the feature vector and bandgap
                feature_vector = X_batch[i].numpy()
                bandgap = y_batch[i].item()
                
                print(f"\nExample {i+1}:")
                print(f"Formula: {original_row['Name']}")
                print(f"Bandgap: {bandgap:.4f}")
                print(f"Feature vector: {np.array2string(feature_vector, precision=4, separator=', ')}")
                
                # Optional: Print more detailed information about the sites if available
                if 'R_a' in original_row and 'R_b' in original_row and 'R_x' in original_row:
                    print(f"Radii - A-site: {original_row['R_a']:.4f}, B-site: {original_row['R_b']:.4f}, X-site: {original_row['R_x']:.4f}")
                
                if 'Tolerance_Factor' in original_row:
                    print(f"Tolerance Factor: {original_row['Tolerance_Factor']:.4f}")
                
                if 'Structure_of_Material' in original_row:
                    print(f"Structure: {original_row['Structure_of_Material']}")
        
        if batch_idx >= 1:  # Only show the first batch for detailed inspection
            break
    
    # Calculate some statistics

    # Modify the dataset iteration part to find NaN values
    all_bandgaps = []
    nan_indices = []
    nan_formulas = []

    # First pass to identify NaN values
    for batch_idx, (X_batch, y_batch) in enumerate(data_loader):
        # Convert to numpy
        y_numpy = y_batch.numpy()
        
        # Check for NaN values in this batch
        for i in range(len(y_numpy)):
            if np.isnan(y_numpy[i]):
                # Calculate dataset index
                dataset_idx = batch_idx * data_loader.dataloader.batch_size + i
                
                if dataset_idx < len(data_loader.dataset):
                    # Get original index and row
                    original_idx = data_loader.dataset.valid_indices[dataset_idx]
                    original_row = data_loader.dataset.df.iloc[original_idx]
                    
                    nan_indices.append(original_idx)
                    nan_formulas.append(original_row['Name'])
                    
                    print(f"NaN found at batch {batch_idx+1}, item {i}, dataset index {dataset_idx}")
                    print(f"  Formula: {original_row['Name']}")
                    print(f"  Original row BandGap value: {original_row['BandGap']}")
                    print(f"  Row data: {original_row[['Name', 'BandGap', 'R_a', 'R_b', 'R_x', 'Tolerance_Factor']].to_dict()}")
            
        # Add all values to list
        all_bandgaps.extend(y_numpy.tolist())

    # Convert to numpy array for analysis
    all_bandgaps = np.array(all_bandgaps)

    # Print NaN summary
    print(f"\nFound {len(nan_indices)} NaN values out of {len(all_bandgaps)} samples")
    print(f"NaN Indices: {nan_indices}")
    print(f"NaN Formulas: {nan_formulas}")

    # Calculate statistics on non-NaN values
    valid_bandgaps = all_bandgaps[~np.isnan(all_bandgaps)]
    print(f"\nDataset processed successfully with {len(data_loader.dataset)} valid samples")
    print(f"Bandgap statistics (excluding NaN values):")
    print(f"  Valid bandgap count: {len(valid_bandgaps)}/{len(all_bandgaps)}")
    print(f"  Min: {np.min(valid_bandgaps):.4f}")
    print(f"  Max: {np.max(valid_bandgaps):.4f}")
    print(f"  Mean: {np.mean(valid_bandgaps):.4f}")
    print(f"  Median: {np.median(valid_bandgaps):.4f}")
    print(f"  Std Dev: {np.std(valid_bandgaps):.4f}")