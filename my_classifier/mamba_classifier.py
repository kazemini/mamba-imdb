import torch
from mamba_ssm import Mamba3
from torch import nn


class MambaClassifier(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim=128, d_state=32, headdim=32, num_classes=2
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.mamba = Mamba3(
            d_model=embedding_dim,
            d_state=d_state,
            headdim=headdim,
            is_mimo=False,
            dtype=torch.float32,
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

        def forward(self, x, lenghts):

            # [B,S] -> [B,S,D]
            embedded = self.embedding(x)

            # [B,S,D] -> [B,S,D]
            features = self.mamba(embedded, lenghts)

            batch_size = features.size(0)

            last_indices = lenghts - 1 # element-wise!

            back_indices = torch.arange(batch_size, device=features.device)

            final_features = features[back_indices, last_indices]

            # [B,D] -> [B,C]
            logits = self.classifier(final_features)
            return logits
