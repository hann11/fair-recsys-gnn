import numpy as np


class ALS:
    def __init__(self, R, k=3, a=40, lambd=10):
        self.R = R
        self.k = k
        self.a = a
        self.lambd = lambd
        self.m, self.n = R.shape
        self.P = np.random.rand(self.m, self.k)
        self.Q = np.random.rand(self.n, self.k)
        self.Phi = np.where(R > 0, 1, 0)
        self.C = 1 + a * R
        self.loss = []

    def train(self, n_step=10):
        l2_reg = self.lambd * np.identity(self.k)
        for step in range(n_step):
            PtP = self.P.T.dot(self.P)
            for i in range(self.n):
                Ci = np.diag(self.C[:, i])
                Wi = (
                    PtP
                    + self.P.T.dot(Ci - np.identity(self.m)).dot(self.P)
                    + l2_reg
                )
                self.Q[i] = np.linalg.inv(Wi).dot(
                    self.P.T.dot(Ci).dot(self.Phi[:, i])
                )

            QtQ = self.Q.T.dot(self.Q)
            for u in range(self.m):
                Cu = np.diag(self.C[u, :])
                Wu = (
                    QtQ
                    + self.Q.T.dot(Cu - np.identity(self.n)).dot(self.Q)
                    + l2_reg
                )
                self.P[u] = np.linalg.inv(Wu).dot(
                    self.Q.T.dot(Cu).dot(self.Phi[u, :])
                )

            _loss = (self.C * (self.Phi - self.P.dot(self.Q.T)) ** 2).sum()
            _l2 = pow(self.P, 2).sum() + pow(self.Q, 2).sum()
            self.loss.append(_loss + self.lambd * _l2)

    def predict(self, u, i):
        return self.P[u].dot(self.Q[i])

    def explain(self, u, i):
        self.P[u]
        qi = self.Q[i]
        Wu = self._Wu(u)
        Cu = np.diag(self.C[u])
        decomp = qi.T.dot(Wu).dot(self.Q.T)
        decomp.dot(Cu).dot(self.Phi[u])

        for i, (sim, conf, phi) in enumerate(
            zip(decomp, Cu.diagonal(), self.Phi[u])
        ):
            print(
                "{:8} | {:30} | {:17} | {:3}".format(
                    i + 1, np.round(sim, 5), conf, phi
                )
            )

    def _Wu(self, u):
        Cu = np.diag(self.C[u, :])
        Wu = (
            self.Q.T.dot(self.Q)
            + self.Q.T.dot(Cu - np.identity(self.n)).dot(self.Q)
            + self.lambd * np.identity(self.k)
        )
        return np.linalg.inv(Wu)
