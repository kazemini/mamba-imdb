import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        self.model.train()

        total_loss = 0
        correct = 0

        # Total number of samples in the dataset
        total = len(self.train_loader.dataset)

        progress_bar = tqdm(self.train_loader, desc="Training", leave=True)

        for batch in progress_bar:
            # Get data from the batch
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            lengths = batch["lengths"].to(self.device)

            # Clear gradients from the previous step
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(input_ids, lengths)

            # Calculate loss
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Update model parameters
            self.optimizer.step()

            # Statistics
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)  # (batch, logit)

            correct += (predictions == labels).sum().item()

        accuracy = correct / total
        average_loss = total_loss / total

        return average_loss, accuracy

    def evaluate(self, data_loader):
        self.model.eval()

        total_loss = 0.0
        correct = 0.0

        # Total number of samples in the dataset
        total = len(data_loader.dataset)
        progress_bar = tqdm(data_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                lengths = batch["lengths"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward only!
                logits = self.model(input_ids, lengths)

                # Loss
                loss = self.criterion(logits, labels)

                # Statistics
                batch_size = labels.size(0)

                total_loss += loss.item() * batch_size

                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()

            average_loss = total_loss / total
            accuracy = correct / total

            return average_loss, accuracy
