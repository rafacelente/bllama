from torch.utils.data import Dataset
import torch
from typing import Optional

class ShakespeareDataset(Dataset):
    def __init__(self, entry: str, tokenizer: object, max_length: Optional[str]=1024):
        self.entry = entry
        self.inputs = []
        self.labels = []
        self.max_length = max_length
        self.tokenizer = tokenizer

    @classmethod
    def from_file(cls, file_path: str, tokenizer: object, max_length: Optional[int] =1024):
        with open(file_path, "r") as f:
            entry = f.read()
        dataset = cls(entry, tokenizer, max_length)
        dataset._prepare_text()
        return dataset
    
    def _prepare_text(self):
        tokenized_text = self.tokenizer.encode(self.entry)

        for i in range(0, len(tokenized_text), self.max_length):
            a = tokenized_text[i:i+self.max_length]
            b = tokenized_text[i+1:i+self.max_length+1]
            if len(a) != self.max_length:
                print(f"{i} this has a different length: {len(a)}, padding") 
                a = a + [0 for _ in range(self.max_length - len(a))]
                b = b + [0 for _ in range(self.max_length - len(b))]
            self.inputs.append(a)
            self.labels.append(b)
            
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx])