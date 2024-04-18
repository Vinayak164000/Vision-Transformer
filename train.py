import torch
from tqdm import tqdm

seed = 42
torch.manual_seed(seed)


class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, optimizer, loss_fn,
                 epochs=5):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs

        self.results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }

    def train_step(self, dataloader):
        self.model.train()
        total_loss, total_acc = 0, 0

        for batch, (X, y) in enumerate(dataloader):
            self.optimizer.zero_grad()

            # Forward pass
            y_pred = self.model(X)

            y = y.type(torch.LongTensor)
            # Calculate loss
            loss = self.loss_fn(y_pred, y)
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Calculate accuracy
            y_pred_class = torch.argmax(y_pred, dim=1)
            total_acc += (y_pred_class == y).sum().item() / len(y_pred)

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        return avg_loss, avg_acc

    def test_step(self, dataloader):
        self.model.eval()
        total_loss, total_acc = 0, 0

        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                test_pred_logits = self.model(X)

                # Calculate loss
                loss = self.loss_fn(test_pred_logits, y)
                total_loss += loss.item()

                # Calculate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                total_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        return avg_loss, avg_acc

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            train_loss, train_acc = self.train_step(self.train_dataloader)
            test_loss, test_acc = self.test_step(self.test_dataloader)

            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

            self.results["train_loss"].append(train_loss)
            self.results["train_acc"].append(train_acc)
            self.results["test_loss"].append(test_loss)
            self.results["test_acc"].append(test_acc)

