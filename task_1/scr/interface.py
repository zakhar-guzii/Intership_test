from abs import ABC, abstructmethod

class MnistClassifierInterface(ABC):
    @abstructmethod
    def train(self, X_train, y_train):
        pass

    @abstructmethod
    def predict(self, X_test):
        pass