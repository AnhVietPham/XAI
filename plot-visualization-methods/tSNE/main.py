import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.datasets import load_digits


def pairwise_distance(X):
    return np.sum((X[None, :] - X[:, None]) ** 2, 2)


def p_conditional(dists, sigmas):
    e = np.exp(-dists / (2 * np.square(sigmas.reshape((-1, 1)))))
    np.fill_diagonal(e, 0.)
    e += 1e-8
    return e / e.sum(axis=1).reshape([-1, 1])


def perp(condi_matr):
    ent = -np.sum(condi_matr * np.log2(condi_matr), 1)
    return 2 ** ent


def search(func, goal, tol=1e-10, max_iters=1000, lowb=1e-20, uppb=1000):
    for _ in range(max_iters):
        guess = (uppb + lowb) / 2
        val = func(guess)

        if val > goal:
            uppb = guess
        else:
            lowb = guess

        if np.abs(val - goal) <= tol:
            return guess
    return guess


def find_sigmas(dists, perplexity):
    found_sigmas = np.zeros(dists.shape[0])
    for i in range(dists.shape[0]):
        func = lambda sig: perp(p_conditional(dists[i: i + 1, :], np.array([sig])))
        found_sigmas[i] = search(func, perplexity)
    return found_sigmas


def q_joint(y):
    dists = pairwise_distance(y)
    nom = 1 / (1 + dists)
    np.fill_diagonal(nom, 0.)
    return nom / np.sum(np.sum(nom))


def gradient(P, Q, y):
    (n, no_dims) = y.shape
    pq_diff = P - Q
    y_diff = np.expand_dims(y, 1) - np.expand_dims(y, 0)
    dists = pairwise_distance(y)

    aux = 1 / (1 + dists)
    return 4 * (np.expand_dims(pq_diff, 2) * y_diff * np.expand_dims(aux, 2)).sum(1)


def m(t):
    return 0.5 if t < 250 else 0.8


def p_joint(X, perp):
    N = X.shape[0]
    dists = pairwise_distance(X)
    sigmas = find_sigmas(dists, perp)
    p_cond = p_conditional(dists, sigmas)
    return (p_cond + p_cond.T) / (2. * N)


def tsne(X, ydim=2, T=1000, l=500, perp=30):
    N = X.shape[0]
    P = p_joint(X, perp)

    Y = []
    y = np.random.normal(loc=0.0, scale=1e-4, size=(N, ydim))
    Y.append(y)
    for t in range(T):
        Q = q_joint(Y[-1])
        grad = gradient(P, Q, Y[-1])
        y = Y[-1] - l * grad + m(t) * (Y[-1] - Y[-2])
        Y.append(y)
        if t % 10 == 0:
            Q = np.maximum(Q, 1e-12)
    return y


if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    test = np.array([
        [1, 2, 3], [4, 5, 6], [7, 8, 9]
    ])
    print(test)
    print(pairwise_distance(test))
    # print(X.shape)
    # print(y)
    # res = tsne(X, T=1000, l=200, perp=40)
    # plt.scatter(X, y, s=20, c=y)
    # plt.show()
