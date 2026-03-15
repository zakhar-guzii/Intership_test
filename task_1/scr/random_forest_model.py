from sklearn.ensemble import RandomForestClassifier

from task_1.scr.interface import MnistClassifierInterface


class RandomForestModel(MnistClassifierInterface):
    def __init__(self):
        self._model = RandomForestClassifier(n_estimators=100)

    def train(self, X_train, y_train):
        X_flat_train = X_train.reshape(len(X_train), -1) / 255.0
        self._model.fit(X_flat_train, y_train)

    def predict(self, X_test):
        X_flat_test = X_test.reshape(len(X_test), -1) / 255.0
        return self._model.predict(X_flat_test)