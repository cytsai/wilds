import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression as Ref
import time
import pickle

DEBUG = 0


class IRLS:
    def __init__(self, n_classes, weight_decay=0.01, max_iter=15):
        self.C = n_classes
        self.weight_decay = weight_decay
        self.max_iter = max_iter

    def _solve(self, H, g):
        try:
            g = torch.cholesky_solve(g.unsqueeze(-1), torch.linalg.cholesky(H))
            return g.squeeze(-1), ''
        except Exception as err:
            g = torch.zeros_like(g)
            return g, err

    def _activation(self, l):
        #return F.softmax(l, dim=-1)
        return l.sigmoid()

    def _fit(self, X, y, device='cuda'):
        X = X.to(device=device)
        y = y.to(device=device)
        D = X.shape[1]
        t = F.one_hot(y, self.C).float()

        W = X.new_zeros(self.C, D)
        H = X.new_zeros(self.C, D, D)
        p = self._activation(X @ W.T)

        for i in range(self.max_iter):
            st = time.time()
            print('-' * 80)
            print('Iteration:', i)

            g = (p - t).T @ X + W * self.weight_decay
            r = p * (1 - p)
            for c in range(self.C):
                Hc = X.T @ (r[:,c].unsqueeze(-1) * X)
                Hc.diag().add_(self.weight_decay)
                H[c] = Hc

            d, err = self._solve(H, g)
            if err:
                print(err)
                break
            W -= d
            #print(f'd_max, W_max = {d.abs().max():.3f}, {W.abs().max():.3f}')
            assert not W.isnan().any()

            l = X @ W.T
            p = self._activation(l)

            BCE = F.binary_cross_entropy(p, t).item() * self.C
            CE  = F.cross_entropy(l, y).item()
            Acc = (l.max(-1)[-1] == y).sum().item() / len(y)
            print(f'BCE, CE, Acc = {BCE:.3f}, {CE:.3f}, {Acc:.3f}')
            print('Time:', time.time()-st)
        #del X, y, t, H
        print('-' * 80)
        return W.cpu()

    def fit(self, X, y):
        self._s, self._u = torch.std_mean(X, dim=0, unbiased=False, keepdim=True)
        X = (X - self._u) / self._s

        D = X.shape[1]
        X = torch.cat((X, X.new_ones(X.shape[0], 1)), dim=-1)
        self.W, self.b = self._fit(X, y).split([D,1], dim=-1)

        self.b -= self.W @ (self._u / self._s).T
        self.W /= self._s
        self.b.squeeze_(-1)

    def score(self, X, y):
        l = X @ self.W.T + self.b
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
        #del X, y

    def get_Wb(self):
        if isinstance(self.clf, IRLS):
            return self.clf.W, self.clf.b

        W = torch.from_numpy(self.clf.coef_)
        b = torch.from_numpy(self.clf.intercept_)
        if self.n_classes > 2:
            return W, b
        else:
            return torch.cat((-W, W)), torch.cat((-b, b))


def _fit(dataset='iwildcam'):
    with open(dataset+'.pkl', 'rb') as f:
        X, y = pickle.load(f)
    clf = LogisticRegression(0, y.max().item()+1)
    clf.fit(X, y)
    return clf.get_Wb()


if __name__ == '__main__':
    dataset = ['iwildcam', 'camelyon17']
    print(_fit(dataset[0]))
