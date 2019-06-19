#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#
# Description: Create matrics for prediciton

import pandas as pd
import numpy as np
import misim


MTREEFILE = "data/mtrees2019.bin"


def getMSN(trainset):
    ''' Accept a training set and retuen a mirna similarity network
    '''
    print("generating mirna similarity network...")
    mtree = misim.loadMtree(MTREEFILE)
    names = sorted(trainset['mir'].unique().tolist())
    mird = {mir:trainset.loc[trainset['mir']==mir, 'disease'].tolist() for mir in names}
    print("total %d mirnas..." % len(names))
    msn = pd.DataFrame(index=names, columns=names)
    for i,x in enumerate(names):
        # print(i)
        for y in names:
            if x == y:
                msn.loc[x, y] = 1.0
            elif pd.isnull(msn.loc[y, x]):
                msn.loc[x, y] = misim.misim(mird[x], mird[y], mtree, 0.5)
            else:
                msn.loc[x, y] = msn.loc[y, x]
    return msn.index.values, msn.values


def getDSN(trainset):
    ''' Accept a training set and return a disease similarity network
    '''
    print("generating disease similarity network...")
    mtree = misim.loadMtree(MTREEFILE)
    names = sorted(trainset['disease'].unique().tolist())
    print("total %d diseases..." % len(names))
    dsn = pd.DataFrame(index=names, columns=names)
    for i,x in enumerate(names):
        # print(i)
        for y in names:
            if x == y:
                dsn.loc[x, y] = 1.0
            elif pd.isnull(dsn.loc[y, x]):
                dagx = misim.genDAG(mtree[x])
                dagy = misim.genDAG(mtree[y])
                dsn.loc[x, y] = misim.semSim(dagx, dagy, 0.5)
            else:
                dsn.loc[x, y] = dsn.loc[y, x]
    return dsn.index.values, dsn.values


def gaussianKS(trainset, gamma=1.0):
    ''' Accept trainset and return gaussian kernel similarity of 
        mirna and disease
    '''
    print("calculating gaussian interaction profile kernel similarity...")
    all_mirna = sorted(trainset['mir'].unique().tolist())
    all_disease = sorted(trainset['disease'].unique().tolist())
    trainset = trainset[['mir', 'disease']]
    adjmat = pd.DataFrame(index=all_mirna, columns=all_disease).fillna(0.0)
    for mir,disease in trainset.values:
        adjmat.loc[mir, disease] = 1.0

    adjmat = adjmat.values

    def calc(adjmat, g):
        row_n = adjmat.shape[0]
        gammam = gamma * row_n / sum([sum(adjmat[i,:]**2) for i in range(row_n)])
        gks = np.zeros((row_n, row_n))
        for i in range(row_n):
            for j in range(row_n):
                if i == j:
                    gks[i, j] = 1.0
                elif i < j:
                    gks[i, j] = np.exp((-1) * gammam * sum((adjmat[i,:] - adjmat[j,:])**2))
                else:
                    gks[i, j] = gks[j, i]
        return gks

    GM = calc(adjmat, gamma)
    GD = calc(adjmat.T, gamma)

    return GM, GD


def saveMatrics(trainset, outdir):

    all_mirna, mfs = getMSN(trainset)
    np.savetxt("%s/all_mirna.txt" % outdir, all_mirna, delimiter="\n", fmt="%s")
    np.savetxt("%s/mirna_funcsim.txt" % outdir, mfs, delimiter="\t", fmt="%.8f")

    all_disease, dss = getDSN(trainset)
    np.savetxt("%s/all_disease.txt" % outdir, all_disease, delimiter="\n", fmt="%s")
    np.savetxt("%s/disease_semsim.txt" % outdir, dss, delimiter="\t", fmt="%.8f")

    GM, GD = gaussianKS(trainset)
    np.savetxt("%s/gaussian_m.txt" % outdir, GM, delimiter="\t", fmt="%.8f")
    np.savetxt("%s/gaussian_d.txt" % outdir, GD, delimiter="\t", fmt="%.8f")

    trainset['val'] = 1
    adjmat = trainset.pivot(index='mir', columns='disease', values='val').fillna(0)
    np.savetxt("%s/adjacent_matrix.txt" % outdir, adjmat.values, delimiter="\t", fmt="%d")

    return 0


