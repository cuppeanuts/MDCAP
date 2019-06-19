#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#
# Predict Causative Association based on Label propagation.

import pandas as pd
import numpy as np
import networkx as nx


def normalization(adjacent_matrix):
    ''' Accept a matrix and return a column normalized matrix
    '''
    am = adjacent_matrix.astype('float')
    asum = am.sum(axis=0)
    asum[asum==0] = 1
    norm_mat = am/asum
    return norm_mat


def matrixCompletion(mfs, dss, gm, gd, adjmat):
    ''' Complete transition matrix with gaussian interaction 
        profile kernels and hub promoted index.
    '''
    # gaussian
    for i in range(len(gm)):
        for j in range(len(gm)):
            if mfs[i,j] == 0:
                mfs[i,j] = gm[i,j]

    for i in range(len(gd)):
        for j in range(len(gd)):
            if dss[i,j] == 0:
                dss[i,j] = gd[i,j]
    
    # hpi
    dd = sum(adjmat)
    md = sum(adjmat.T)

    for i in range(len(mfs)):
        for j in range(len(mfs)):
            a = adjmat[i,:]
            b = adjmat[j,:]
            s = sum(a[(a+b)==2])
            hpi = 1.0 + float(s)/min(md[i], md[j])
            mfs[i, j] *= hpi*max(md[i], md[j])

    for i in range(len(dss)):
        for j in range(len(dss)):
            a = adjmat[:,i]
            b = adjmat[:,j]
            s = sum(a[(a+b)==2])
            hpi = 1.0 + float(s)/min(dd[i], dd[j])
            dss[i, j] *= hpi*max(dd[i], dd[j])

    return mfs, dss


def dualLP(transmat_m, transmat_d, adjmat, r=0.5, beta=0.6):

    # prediction
    adjmat = adjmat.astype(float)

    M = adjmat
    D = adjmat.T

    for i in range(len(transmat_m)):
            transmat_m[i, i] = 0.0
    for i in range(len(transmat_d)):
            transmat_d[i, i] = 0.0

    TM = normalization(transmat_m)
    TD = normalization(transmat_d)

    loss = 1e-10

    delta = 1.0
    M0 = M
    while delta > loss:
            M_ = (1.0-r)*np.dot(TM, M) + r*M0
            delta = abs(sum(sum(abs(M_ - M))))
            M = M_

    delta = 1.0
    D0 = D
    while delta > loss:
            D_ = (1.0-r)*np.dot(TD, D) + r*D0
            delta = abs(sum(sum(abs(D_ - D))))
            D = D_

    y_ = beta*M_ + (1-beta)*(D_.T)

    return y_


def main(outdir):

    print("reading files from %s ..." % outdir)
    trainset = pd.read_csv("%s/trainset.txt" % outdir, sep="\t", index_col=[0,1])
    testset = pd.read_csv("%s/testset.txt" % outdir, sep="\t", index_col=[0,1])
    adjmat = np.loadtxt("%s/adjacent_matrix.txt" % outdir, delimiter="\t")
    all_mirna = np.loadtxt("%s/all_mirna.txt" % outdir, dtype=str, delimiter="\n")
    all_disease = np.loadtxt("%s/all_disease.txt" % outdir, dtype=str, delimiter="\n")
    mfs = np.loadtxt("%s/mirna_funcsim.txt" % outdir, delimiter="\t")
    dss = np.loadtxt("%s/disease_semsim.txt" % outdir, delimiter="\t")
    gm = np.loadtxt("%s/gaussian_m.txt" % outdir, delimiter="\t")
    gd = np.loadtxt("%s/gaussian_d.txt" % outdir, delimiter="\t")
    
    print("completing transition matrix ...")
    transmat_m, transmat_d = matrixCompletion(mfs, dss, gm, gd, adjmat)

    print("predicting by dual label propagation ...")
    y = dualLP(transmat_m, transmat_d, adjmat)

    print("collecting results ...")
    y = pd.DataFrame(y, index=all_mirna, columns=all_disease)
    y = pd.DataFrame(y.stack())
    y['cond'] = 0
    y.loc[testset.index, 'cond'] = 1
    y = y.drop(trainset.index)
    y = y.reset_index()
    y.columns = ['mir', 'disease', 'pred', 'cond']

    y.to_csv("%s/dlp_results.txt" % outdir, sep="\t", index=False)
    
    print("finished.")

    return y

if __name__ == '__main__':

    outdir = "./inputdata/causal"
    main(outdir)
