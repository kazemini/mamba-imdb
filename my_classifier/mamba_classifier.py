import torch
from mamba_ssm import Mamba3
from torch import nn


class MambaClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        d_state=64,
        headdim=32,
        num_classes=2,
        dropout=0.2,
        pad_token_id=0
    ):
        super().__init__()

        self.pad_token_id = pad_token_id

        # Token embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_token_id
        )

        # Mamba-3
        self.mamba = Mamba3(
            d_model=embedding_dim,
            d_state=d_state,
            headdim=headdim,
            is_mimo=False,
            is_outproj_norm=True,
            dtype=torch.float32
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths):

        # [B,S] → [B,S,D]
        embedded = self.embedding(x)

        # [B,S,D]
        features = self.mamba(embedded)

        # Create mask for real tokens
        batch_size, seq_len = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device
        ).unsqueeze(0)

        mask = positions < lengths.unsqueeze(1)

        # [B,S,1]
        mask = mask.unsqueeze(-1)

        # Remove padding representations
        masked_features = features * mask

        # Sum real token representations
        summed_features = masked_features.sum(dim=1)

        # Number of real tokens
        lengths = lengths.clamp(min=1).unsqueeze(-1)

        # Mean pooling
        pooled_features = summed_features / lengths

        # [B,128] → [B,2]
        logits = self.classifier(pooled_features)

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
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():

            logits = self(
                input_tensor,
                lengths
            )

            predicted_class = torch.argmax(
                logits,
                dim=1
            ).item()

        return (
            "Positive"
            if predicted_class == 1
            else "Negative"
        )