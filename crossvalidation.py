#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#
# Cross Validation


import os
import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import misim
import duallp
import matrics
import curves


def nfoldCV(dataset, n=5):
    totalnum = len(dataset)
    partsize = math.ceil(totalnum/float(n))
    idx = set(dataset.index.values.tolist())
    rest = set(list(idx))
    indexlist = []
    while len(rest)//partsize != 0:
        testsetidx = set(random.sample(rest, int(partsize)))
        trainsetidx = idx - testsetidx
        rest -= testsetidx
        indexlist.append([testsetidx, trainsetidx])
    if rest:
        indexlist.append([rest, idx-rest])
    for testset, trainset in indexlist:
        yield (dataset.loc[testset], dataset.loc[trainset])

def getDataset():

    dataset = pd.read_csv("data/hmdd_causal.txt", sep="\t")

    random.seed(1024)
    idx = set(dataset.index.values.tolist())
    testsetidx = set(random.sample(idx, len(dataset)//5))
    trainsetidx = idx -  testsetidx

    testset = dataset.loc[testsetidx]
    trainset = dataset.loc[trainsetidx]

    testset.to_csv("data/hmdd_causal_test.txt", sep="\t", index=False)
    trainset.to_csv("data/hmdd_causal_train.txt", sep="\t", index=False)

    return 0

def crossvalidation():
    
    dataset = pd.read_csv("data/hmdd_causal_train.txt", sep="\t")
   
    n = 10
    random.seed(1024)
    i = 0

    for testset, trainset in nfoldCV(dataset, n):
        
        i += 1
        outdir = "inputdata/crossvalidation/data_%02d" % i
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        testset = testset[np.in1d(testset['mir'], trainset['mir'])]
        testset = testset[np.in1d(testset['disease'], trainset['disease'])]
        testset.to_csv("%s/testset.txt" % outdir, sep="\t", index=False)
        trainset.to_csv("%s/trainset.txt" % outdir, sep="\t", index=False)
        
        matrics.saveMatrics(trainset, outdir)

        duallp.main(outdir)

    results = pd.DataFrame(columns=["mir", "disease", "pred", "cond"])
    for i in range(10):
        i += 1
        outdir = "inputdata/crossvalidation/data_%02d" % i
        subres = pd.read_csv("%s/dlp_results.txt" % outdir, sep="\t")
        results = pd.concat([results, subres], axis=0)
    results.to_csv("results/cv_results.txt", sep="\t", index=False)

    return 0

def test():

    trainset = pd.read_csv("data/hmdd_causal_train.txt", sep="\t")
    testset = pd.read_csv("data/hmdd_causal_test.txt", sep="\t")

    outdir = "inputdata/cv_test"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    testset = testset[np.in1d(testset['mir'], trainset['mir'])]
    testset = testset[np.in1d(testset['disease'], trainset['disease'])]
    testset.to_csv("%s/testset.txt" % outdir, sep="\t", index=False)
    trainset.to_csv("%s/trainset.txt" % outdir, sep="\t", index=False)

    matrics.saveMatrics(trainset, outdir)
    
    duallp.main(outdir)

    results = pd.read_csv("%s/dlp_results.txt" % outdir, sep="\t")
    results.to_csv("results/cv_test.txt", sep="\t")

    return 0
    

def showCurves():
    
    results_cv = pd.read_csv("results/cv_results.txt", sep="\t")
    results_cv['cond'] = results_cv['cond'].astype(float)
    results = pd.read_csv("results/cv_test.txt", sep="\t")
    results['cond'] = results['cond'].astype(float)
    # draw curves
    fig, sp = plt.subplots(1, 2, figsize=(8,4))
    curves.drawROC(results_cv, sp[0], "MDCAP cv")
    curves.drawROC(results, sp[1], "MDCAP cv_test")
    plt.show()


if __name__ == '__main__':
    #getDataset()
    #crossvalidation()
    #test()
    showCurves()


