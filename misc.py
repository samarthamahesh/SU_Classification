#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random

from data_import import AdultDataset


def gen1(n, dim, mean=2, var=1):
    return np.random.normal(mean, var, size=(n, dim))


def gen0(n, dim, mean=-2, var=1):
    return np.random.normal(mean, var, size=(n, dim))


# def synth_dataset(ns, nu, prior, dim=2):
#     nsp = np.random.binomial(ns, prior**2 / (prior**2 + (1-prior)**2))
#     nsn = ns - nsp
#     xs = np.concatenate((
#         np.hstack((gen1(nsp, dim), gen1(nsp, dim))),
#         np.hstack((gen0(nsn, dim), gen0(nsn, dim)))))

#     nup = np.random.binomial(nu, prior)
#     nun = nu - nup
#     xu = np.concatenate((gen1(nup, dim), gen0(nun, dim)))

#     return xs, xu

def synth_dataset(ds, ns, nd, prior):
    nspp = np.random.binomial(ns, prior**2 / (prior**2 + (1-prior)**2))
    nsnn = ns - nspp
    xs = np.concatenate((
        np.hstack((ds.get_gen1(nspp), ds.get_gen1(nspp))),
        np.hstack((ds.get_gen0(nsnn), ds.get_gen0(nsnn)))))


    ndpn = np.random.binomial(nd, 0.5)
    ndnp = nd - ndpn
    xd = np.concatenate((
        np.hstack((ds.get_gen1(ndpn), ds.get_gen0(ndpn))),
        np.hstack((ds.get_gen0(ndnp), ds.get_gen1(ndnp)))))

    return xs, xd


def synth_dataset_test(ds, n, prior):
    n1 = np.random.binomial(n, prior)
    n0 = n - n1
    x = np.concatenate((ds.get_gen1(n1), ds.get_gen0(n0)))
    y = np.concatenate((np.ones(n1), -np.ones(n0)))
    return x, y


# def load_dataset(n_s, n_u, n_test, prior, dim=2):
#     x_s, x_u = synth_dataset(n_s, n_u, prior, dim)
#     x_test, y_test = synth_dataset_test(n_test, prior, dim)
#     return x_s, x_u, x_test, y_test


# def convert_su_data_sklearn_compatible(x_s, x_u):
#     x = np.concatenate((x_s.reshape(-1, x_s.shape[1] // 2), x_u))
#     y = np.concatenate((np.ones(x_s.shape[0] * 2), np.zeros(x_u.shape[0])))
#     return x, y

def load_dataset(n_s, n_d, n_test, prior):
    dataset = AdultDataset()
    x_s, x_d = synth_dataset(dataset, n_s, n_d, prior)
    x_test, y_test = synth_dataset_test(dataset, n_test, prior)
    return x_s, x_d, x_test, y_test


def convert_su_data_sklearn_compatible(x_s, x_d, noise):
    x = np.concatenate((x_s.reshape(-1, x_s.shape[1] // 2), x_d.reshape(-1, x_d.shape[1] // 2)))
    
    y1 = np.ones(x_s.shape[0] * 2)
    ny1 = len(y1)
    nind = random.sample(list(range(ny1)), int(ny1*noise))
    y1[nind] = 0

    y0 = np.zeros(x_d.shape[0] * 2)
    ny0 = len(y0)
    nind = random.sample(list(range(ny0)), int(ny0*noise))
    y0[nind] = 1
    
    y = np.concatenate((y1, y0))
    return x, y
