import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from my_tokenizers.base_tokenizer import BaseTokenizer


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_classes=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):

        # x: Token IDs
        # Shape: [batch_size, sequence_length]

        embedded = self.embedding(x)

        # embedded:
        # Shape: [batch_size, sequence_length, embedding_dim]

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (h_n, _) = self.lstm(packed)

        # h_n:
        # Shape: [num_layers, batch_size, hidden_dim]

        final_hidden_state = h_n[-1]

        # Shape: [batch_size, hidden_dim]

        logits = self.classifier(final_hidden_state)

        # Shape: [batch_size, num_classes]

        return logits

    def load_weights(self, path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(path, map_location=device)

        self.load_state_dict(state_dict)

        return self

    def predict_sentiment(self, text, tokenizer: BaseTokenizer):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval()

        input_ids = tokenizer.encode(text)

        # Convert input_ids to a tensor and move it to the specified device
        # [1, Sequence Length]
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

        # Length of the input sequence
        # [1]
        length = torch.tensor([len(input_ids)], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = self(input_tensor, length)  # [1, num_classes]
            predicted_class = torch.argmax(logits, dim=1).item()

        return "Positive" if predicted_class == 1 else "Negative"
