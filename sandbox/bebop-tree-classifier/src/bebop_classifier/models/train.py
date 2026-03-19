class Trainer:
    def __init__(self, model, dataset, optimizer, criterion, device):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_model(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            for inputs, labels in self.dataset:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')