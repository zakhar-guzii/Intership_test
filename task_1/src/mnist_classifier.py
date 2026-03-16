from src.cnn_model import ConvolutionNN
from src.ff_nn_model import FeedForwardNN
from src.random_forest_model import RandomForestModel

class MnistClassifier:
    def __init__(self, algorithm: str):
        self.algorihm = algorithm

        if self.algorihm == "rf":
            self._model = RandomForestModel()
        elif self.algorihm == 'nn':
            self._model = FeedForwardNN()
        elif self.algorihm == 'cnn':
            self._model = ConvolutionNN()
        else: 
            raise ValueError(f"Unknown algorithm: {self.algorihm}")

    
    def train(self, X_train, y_train):
        self._model.train(X_train, y_train)

    def predict(self, X_test):
        return self._model.predict(X_test)