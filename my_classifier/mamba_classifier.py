import torch
from mamba_ssm import Mamba3
from torch import nn


class MambaClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        d_state=32,
        headdim=32,
        num_classes=2
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.mamba = Mamba3(
            d_model=embedding_dim,
            d_state=d_state,
            headdim=headdim,
            is_mimo=False,
            dtype=torch.float32
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)

        features = self.mamba(embedded)

        batch_size = features.size(0)

        last_indices = lengths - 1

        batch_indices = torch.arange(
            batch_size,
            device=features.device
        )

        final_features = features[
            batch_indices,
            last_indices
        ]

        logits = self.classifier(final_features)

        return logits

    def load_weights(self, path):
        device = next(self.parameters()).device

        state_dict = torch.load(
            path,
            map_location=device
        )

        self.load_state_dict(state_dict)

        return self


    def predict_sentiment(self, text, tokenizer):
        self.eval()

        device = next(self.parameters()).device

        input_ids = tokenizer.encode(text)

        input_tensor = torch.tensor(
            input_ids,
            dtype=torch.long
        ).unsqueeze(0).to(device)

        lengths = torch.tensor(
            [len(input_ids)],
            dtype=torch.long
        ).to(device)

        with torch.no_grad():
            logits = self(input_tensor, lengths)

            predicted_class = torch.argmax(
                logits,
                dim=1
            ).item()

        return "Positive" if predicted_class == 1 else "Negative"