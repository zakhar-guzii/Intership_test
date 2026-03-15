from task_1.scr.interface import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(MnistClassifierInterface):
    def __init__(self):
        self._model = RandomForestClassifier(n_estimators=100)
    
    def _prepoces(self, X):
        X_norm = X/255
        return X_norm
    
    def train(self, X_train, y_train):
        X_train_norm = self._prepoces(X_train)
        self._model.fit(X_train_norm, y_train)

    def predict(self, X_test):
        X_test_norm = self._prepoces(X_test)
        return self._model.predict(X_test_norm)