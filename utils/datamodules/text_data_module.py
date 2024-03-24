from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def collate_batch(batch, max_length=1024):
    input_ids, labels = zip(*batch)

    max_len = min(max(len(x) for x in input_ids), max_length)
    padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return padded_inputs[:max_len], padded_labels[:max_len]

class TextDataModule(LightningDataModule):
    def __init__(self,
                    dataset: Dataset, 
                    batch_size:int=8,
                    train_test_split:float=0.8,
                    seed:int=42
                ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.seed = seed
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_predict = None

    def setup(self, stage=None):
        if stage == "fit":
            train_size = int(len(self.dataset) * self.train_test_split)
            val_size = len(self.dataset) - train_size
            self.dataset_train, self.dataset_val = random_split(
                self.dataset, 
                [train_size, val_size], 
                generator=torch.Generator().manual_seed(self.seed)
            )
        if stage == "predict":
            self.dataset_predict = self.dataset

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train, 
            batch_size=self.batch_size, 
            shuffle=True, num_workers=7,
            #pin_memory=True, 
            collate_fn=lambda x: collate_batch(x, max_length=self.dataset.max_length)
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset_val, 
            batch_size=self.batch_size, 
            shuffle=False, num_workers=7,
            #pin_memory=True, 
            collate_fn=lambda x: collate_batch(x, max_length=self.dataset.max_length)
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.dataset_predict,
            batch_size=self.batch_size, 
            shuffle=False, num_workers=0, 
            collate_fn=lambda x: collate_batch(x, max_length=self.dataset.max_length)
        )