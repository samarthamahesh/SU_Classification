import argparse
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets

from misc import load_dataset, convert_su_data_sklearn_compatible


class SD_Base(BaseEstimator, ClassifierMixin):

    def __init__(self, prior=.7, lam=1):
        self.prior = prior
        self.lam = lam

    def fit(self, x, y):
        pass

    def predict(self, x):
        check_is_fitted(self, 'coef_')
        x = check_array(x)
        return np.sign(.1 + np.sign(self._basis(x).dot(self.coef_)))

    def score(self, x, y):
        x_s, x_d = x[y == 1, :], x[y == 0, :]
        f = self.predict
        p_p = self.prior
        p_n = 1 - self.prior
        p_s = p_p ** 2 + p_n ** 2
        p_d = 2 * p_p * p_n

        # SD risk estimator with zero-one loss
        r_s = (p_p * (1 - np.sign(f(x_s))) - p_n * (1 - np.sign(-f(x_s)))) * p_s / (p_p - p_n)
        r_d = (p_p * (1 - np.sign(-f(x_d))) - p_n * (1 - np.sign(f(x_d)))) * p_d / (p_p - p_n)
        risk = r_s.mean() + r_d.mean()

        # makes higher score means good performance
        score = np.maximum(0, 1 - risk)
        return score

    def _basis(self, x):
        # linear basis
        return np.hstack((x, np.ones((len(x), 1))))


class SD_SL(SD_Base):

    def fit(self, x, y):
        check_classification_targets(y)
        x, y = check_X_y(x, y)
        x_s, x_d = x[y == +1, :], x[y == 0, :]
        n_s, n_d = len(x_s), len(x_d)

        p_p = self.prior
        p_n = 1 - self.prior
        p_s = p_p ** 2 + p_n ** 2
        p_d = 1 - p_s
        k_s = self._basis(x_s)
        k_d = self._basis(x_d)
        d = k_d.shape[1]

        """
        Note that `2 *` is needed for `b` while this coefficient does not seem
        appear in the original paper at a glance.
        This is because `k_s.T.mean` takes mean over `2 * n_s` entries,
        while the division is taken with `n_s` in the original paper.
        """
        A = p_s * k_s.T.dot(k_s) / n_s + p_d * k_d.T.dot(k_d) / n_d + self.lam * np.eye(d)
        b = (2 * p_s * k_s.T.mean(axis=1) - 2 * p_d * k_d.T.mean(axis=1)) / (2 * (p_p - p_n))
        self.coef_ = np.linalg.solve(A, b)

        return self


class SD_DH(SD_Base):

    def fit(self, x, y):
        from cvxopt import matrix, solvers
        solvers.options['show_progress'] = False

        check_classification_targets(y)
        x, y = check_X_y(x, y)
        x_s, x_d = x[y == +1, :], x[y == 0, :]
        n_s, n_d = len(x_s), len(x_d)

        p_p = self.prior
        p_n = 1 - self.prior
        p_s = p_p ** 2 + p_n ** 2
        k_s = self._basis(x_s)
        k_u = self._basis(x_d)
        d = k_u.shape[1]

        P = np.zeros((d + 2 * n_d, d + 2 * n_d))
        P[:d, :d] = self.lam * np.eye(d)
        q = np.vstack((
            -p_s / (n_s * (p_p - p_n)) * k_s.T.dot(np.ones((n_s, 1))),
            -p_n / (n_d * (p_p - p_n)) * np.ones((n_d, 1)),
            -p_p / (n_d * (p_p - p_n)) * np.ones((n_d, 1))
        ))
        G = np.vstack((
            np.hstack((np.zeros((n_d, d)), -np.eye(n_d), np.zeros((n_d, n_d)))),
            np.hstack((0.5 * k_u, -np.eye(n_d), np.zeros((n_d, n_d)))),
            np.hstack((k_u, -np.eye(n_d), np.zeros((n_d, n_d)))),
            np.hstack((np.zeros((n_d, d)), np.zeros((n_d, n_d)), -np.eye(n_d))),
            np.hstack((-0.5 * k_u, np.zeros((n_d, n_d)), -np.eye(n_d))),
            np.hstack((-k_u, np.zeros((n_d, n_d)), -np.eye(n_d)))
        ))
        h = np.vstack((
            np.zeros((n_d, 1)),
            -0.5 * np.ones((n_d, 1)),
            np.zeros((n_d, 1)),
            np.zeros((n_d, 1)),
            -0.5 * np.ones((n_d, 1)),
            np.zeros((n_d, 1))
        ))
        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        self.coef_ = np.array(sol['x'])[:d]


def class_prior_estimation(DS, DU):
    # class-prior estimation using MPE method in Ramaswamy et al. (2016)
    from mpe import wrapper
    km1, km2 = wrapper(DU, DS.reshape(-1, DS.shape[1]//2))
    prior_p = km2
    return 0.5 * (np.sqrt(2 * prior_p - 1) + 1)


def main(loss_name, prior=0.7, noise=0.0, n_s=500, n_d=500, end_to_end=False):
    # if loss_name == 'squared':
    #     SD = SD_SL
    # elif loss_name == 'double-hinge':
    #     SD = SD_DH

    SD = SD_SL

    # load dataset
    n_test = 1000
    x_s, x_d, x_test, y_test = load_dataset(n_s, n_d, n_test, prior)
    x_train, y_train = convert_su_data_sklearn_compatible(x_s, x_d, noise)

    if end_to_end:
        # use KM2 (Ramaswamy et al., 2016)
        est_prior = class_prior_estimation(x_s, x_d)
    else:
        # use the pre-fixed class-prior
        est_prior = prior

    # cross-validation
    lam_list = [1e-01, 1e-04, 1e-07]
    score_cv_list = []
    for lam in lam_list:
        clf = SD(prior=est_prior, lam=lam)
        score_cv = cross_val_score(clf, x_train, y_train, cv=5).mean()
        score_cv_list.append(score_cv)

    # training with the best hyperparameter
    lam_best = lam_list[np.argmax(score_cv_list)]
    clf = SD(prior=est_prior, lam=lam_best)
    clf.fit(x_train, y_train)

    # test prediction
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--loss',
        action   = 'store',
        required = False,
        type     = str,
        choices  = ['squared', 'double-hinge'],
        help     = 'loss function')

    parser.add_argument('--ns',
        action   = 'store',
        required = False,
        type     = int,
        default  = 500,
        help     = 'number of similar data pairs')

    parser.add_argument('--nd',
        action   = 'store',
        required = False,
        type     = int,
        default  = 500,
        help     = 'number of unlabeled data points')

    parser.add_argument('--prior',
        action   = 'store',
        required = False,
        type     = float,
        default  = 0.7,
        help     = 'true class-prior (ratio of positive data)')

    parser.add_argument('--noise',
        action   = 'store',
        required = False,
        type     = float,
        default  = 0.0,
        help     = 'noise in pairwise data')

    parser.add_argument('--full',
        action   = 'store_true',
        default  = False,
        help     = 'do end-to-end experiment including class-prior estimation (default: false)')

    args = parser.parse_args()
    main(args.loss, args.prior, args.noise, args.ns, args.nd, args.full)
