from torch.utils.data import Dataset
import torch
from typing import Optional, Union, List
import pandas as pd


class GepetoDataset(Dataset):
    """
    Generic dataset class that can be used to load any parquet file or list of parquet files.
    """
    def __init__(self, 
                file_paths: Union[str, List[str]],
                tokenizer: object, 
                max_length: Optional[str]=1024):
        self.df = None
        self.file_paths = file_paths
        self.max_length = max_length
        self.tokenizer = tokenizer

        self._setup()
        self._tokenize(prune_dataset=True)

    def _setup(self):
        for file_path in self.file_paths:
            assert file_path.endswith(".parquet"), "File path must be a parquet file"
            df = pd.read_parquet(file_path)
            assert "text" in df.columns, "Column 'text' not found in dataframe"
            df = df[["text"]]
            self.df = df if self.df is None else pd.concat([self.df, df])
        self.df.reset_index(drop=True, inplace=True)

    def _tokenize(self, prune_dataset=True):
        print('Tokenizing text...')
        self.df["tokens"] = self.df["text"].apply(lambda x: self.tokenizer.encode(x, allowed_special={'<|endoftext|>'}))
        self.df["len_tokens"] = self.df["tokens"].apply(lambda x: len(x))
        if prune_dataset:
            self.df = self.df[self.df["len_tokens"] > 50]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        tokens = self.df.iloc[idx]["tokens"]
        tokens = tokens[:self.max_length]
        inputs = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return inputs, labels