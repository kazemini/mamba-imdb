import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class IMDBTorchDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        text = row["review"]
        label = row["sentiment"]
        input_ids = self.tokenizer.encode(text)
        
        return {
            "input_ids": input_ids,
            "label": label
        }


def collate_fn(batch, pad_token_id):

    input_ids = [
        sample["input_ids"]
        for sample in batch
    ]

    labels = [
        sample["label"]
        for sample in batch
    ]

    lengths = torch.tensor(
        [len(ids) for ids in input_ids],
        dtype=torch.long
    )

    input_ids = pad_sequence(
        [
            torch.tensor(ids, dtype=torch.long)
            for ids in input_ids
        ],
        batch_first=True,
        padding_value=pad_token_id
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    return {
        "input_ids": input_ids,
        "lengths": lengths,
        "labels": labels
    }