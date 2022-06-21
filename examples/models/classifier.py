import numpy as np
from sklearn.linear_model import LogisticRegression as _LogisticRegression


class LogisticRegression:
    def __init__(self, seed, n_classes):
        self.clf = _LogisticRegression(verbose=1, random_state=seed, solver='saga', C=100, tol=0.01, max_iter=1000, multi_class='multinomial')
        self.n_classes = n_classes
        self.X = []
        self.y = []
        self.use = False
        self.fitted = False

    @property
    def train(self):
        return self.use and not self.fitted

    def update(self, X, y):
        self.X.append(X.cpu())
        self.y.append(y.cpu())

    def fit(self):
        X = np.concatenate(self.X)
        y = np.concatenate(self.y)
        del self.X, self.y
        assert len(set(y)) == self.n_classes

        print('='*80)
        print(X.shape, y.shape)
        self.clf.fit(X, y)
        self.fitted = True
        print('Acc:', self.clf.score(X, y))
        print('='*80)
        del X, y

    def get_Wb(self):
        W, b = self.clf.coef_, self.clf.intercept_
        if len(self.clf.classes_) > 2:
            return W, b
        else:
            return np.concatenate((-W, W)), np.concatenate((-b, b))

