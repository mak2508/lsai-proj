import pyarrow.parquet as pq
from torch.utils.data import Dataset

class ParquetDataset(Dataset):
    def __init__(self, parquet_file: str, tokenizer: str, sequence_length: int, training_samples: int):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.training_samples = training_samples

    def __len__(self):
        return self.training_samples

    def __getitem__(self, idx: int):
        sample_str = str(self.parquet_ds["text"][idx % self.real_length])
        return self.tokenizer.encode_plus(sample_str,
                                          max_length=self.sequence_length + 1,
                                          padding='max_length',
                                          truncation=True,
                                          padding_side="right")


from dataclasses import dataclass
from typing import List, Dict
import torch

@dataclass
class CollatorForCLM:
    sequence_length: int
    pad_token_id: int

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.LongTensor([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s+1)

        inputs = input_ids[:, :-1].clone()
        labels = input_ids[:, 1:]

        # For padding tokens, mask the loss
        labels[labels == self.pad_token_id] = -100

        assert inputs.shape[1] == labels.shape[1] == self.sequence_length
        assert inputs.shape == labels.shape

        return inputs, labels
