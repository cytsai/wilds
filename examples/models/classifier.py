import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression as Ref
import pickle

DEBUG = 0


class IRLS:
    def __init__(self, n_classes, weight_decay=0.01, max_iter=5):
        self.C = n_classes
        self.weight_decay = weight_decay
        self.max_iter = max_iter

    def _solve(self, H, g):
        g = g.unsqueeze(-1)
        LD, pivots, _ = torch.linalg.ldl_factor_ex(H)
        g = torch.linalg.ldl_solve(LD, pivots, g)
        return g.squeeze(-1)

    def _activation(self, l):
        #return F.softmax(l, dim=-1)
        return l.sigmoid()

    def _fit(self, X, y, device='cuda'):
        X = X.to(device=device)
        y = y.to(device=device)
        D = X.shape[1]
        t = F.one_hot(y, self.C).float()

        W = X.new_zeros(D, self.C)
        H = X.new_zeros(self.C, D, D)
        p = self._activation(X @ W)

        for i in range(self.max_iter):
            print('-' * 80)
            print('Iteration:', i)
            g = X.T @ (p - t) + W * self.weight_decay
            r = p * (1 - p)
            for c in range(self.C):
                XTX = X.T @ (r[:,c].unsqueeze(-1) * X)
                XTX.diag().add_(self.weight_decay) # TODO: check alg
                H[c] = XTX

            d  = self._solve(H, g.T).T
            W -= d
            print(f'd_max, W_max = {d.abs().max():.3f}, {W.abs().max():.3f}')

            l = X @ W
            p = self._activation(l)

            BCE = F.binary_cross_entropy(p, t).item() * self.C
            CE  = F.cross_entropy(l, y).item()
            Acc = (l.max(-1)[-1] == y).sum().item() / len(y)
            print(f'BCE, CE, Acc = {BCE:.3f}, {CE:.3f}, {Acc:.3f}')
        print('-' * 80)
        del X, y, t, H
        return W.cpu()

    def fit(self, X, y):
        D = X.shape[1]
        X = torch.cat((X, X.new_ones(X.shape[0], 1)), dim=-1)
        self.W, self.b = self._fit(X, y).split([D,1], dim=0)
        self.b.squeeze_(0)

    def score(self, X, y):
        l = X @ self.W + self.b #.unsqueeze(0)
        return (l.max(-1)[-1] == y).sum().item() / len(y)


class LogisticRegression:
    def __init__(self, seed, n_classes):
        #self.clf = Ref(verbose=1, random_state=seed, solver='saga', C=100, tol=0.01, max_iter=1000, multi_class='multinomial')
        self.clf = IRLS(n_classes)
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

    def fit(self, X=None, y=None):
        X = torch.cat(self.X) if X is None else X
        y = torch.cat(self.y) if y is None else y
        del self.X, self.y
        assert len(y.unique()) == self.n_classes

        if DEBUG:
            with open('features.pkl', 'wb') as f:
                pickle.dump((X, y), f)

        print('=' * 80)
        self.clf.fit(X, y)
        self.fitted = True
        print('Acc:', self.clf.score(X, y))
        print('=' * 80)
        del X, y

    def get_Wb(self):
        if isinstance(self.clf, IRLS):
            return self.clf.W, self.clf.b

        W = torch.from_numpy(self.clf.coef_)
        b = torch.from_numpy(self.clf.intercept_)
        if len(self.n_classes) > 2:
            return W, b
        else:
            return torch.cat((-W, W)), torch.cat((-b, b))


if __name__ == '__main__':
    with open('features.pkl', 'rb') as f:
        X, y = pickle.load(f)
    clf = LogisticRegression(0, y.max().item()+1)
    clf.fit(X, y)
