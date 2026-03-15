import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import logging

from task_1.scr.interface import MnistClassifierInterface


logger = logging.getLogger(__name__)


class FeedForwardNN(MnistClassifierInterface):
    def __init__(self):
        self._model = nn.Sequential(
                nn.Linear(784, 250),
                nn.ReLU(),
                nn.Linear(250, 100),
                nn.ReLU(),
                nn.Linear(100, 10)
            )
        self.epochs = 5
        self.batch_size = 32
        self.learning_rate = 0.001

    def train(self, X_train, y_train):
        tensor_X = torch.Tensor(X_train / 255.0).view(-1, 784)
        tensor_y = torch.LongTensor(y_train)

        dataset = TensorDataset(tensor_X, tensor_y)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self._model.train()

        for epoch in range(self.epochs):
            correct = 0
            total_loss = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self._model(X_batch)
                
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predicted = torch.max(output.data, 1)[1] 
                correct += (predicted == y_batch).sum().item()

            accuracy = 100. * correct / len(train_loader.dataset)
            avg_loss = total_loss / len(train_loader)
            logger.info(f'Epoch [{epoch+1}/{self.epochs}] \t Loss: {avg_loss:.4f} \t Accuracy: {accuracy:.2f}%')

    def predict(self, X_test):
        self._model.eval()
        
        tensor_X = torch.Tensor(X_test / 255.0).view(-1, 784)
        
        with torch.no_grad():
            output = self._model(tensor_X)
            predictions = torch.max(output.data, 1)[1]
            
        return predictions.numpy()