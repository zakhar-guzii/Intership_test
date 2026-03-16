import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import logging

from src.interface import MnistClassifierInterface


logger = logging.getLogger(__name__)

class ConvolutionNN(MnistClassifierInterface):
    def __init__(self):
        self._model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        self.epochs = 5
        self.batch_size = 64
        self.learning_rate = 0.001

    def train(self, X_train, y_train):
        tensor_X = torch.Tensor(X_train / 255.0).view(-1, 1, 28, 28)
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
        tensor_X = torch.Tensor(X_test / 255.0).view(-1, 1, 28, 28)
        
        with torch.no_grad():
            output = self._model(tensor_X)
            predictions = torch.max(output.data, 1)[1]
            
        return predictions.numpy()