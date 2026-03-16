from src.cnn_model import ConvolutionNN
from src.ff_nn_model import FeedForwardNN
from src.random_forest_model import RandomForestModel


ALGORITHM_MAP = {
    "rf": RandomForestModel,
    "nn": FeedForwardNN,
    "cnn": ConvolutionNN,
}

class MnistClassifier:
    def __init__(self, algorithm: str):
        if algorithm not in ALGORITHM_MAP:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {list(ALGORITHM_MAP)}")
        self._model = ALGORITHM_MAP[algorithm]()
    
    def train(self, X_train, y_train):
        self._model.train(X_train, y_train)

    def predict(self, X_test):
        return self._model.predict(X_test)