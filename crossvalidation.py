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


def main():
    
    dataset = pd.read_csv("data/hmdd_causal.txt", sep="\t")
   
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


def showCurves():
   
    results = pd.read_csv("results/cv_results.txt", sep="\t")
    results['cond'] = results['cond'].astype(float)
    # draw curves
    fig, sp = plt.subplots(1, 1, figsize=(4,4))
    #curves.drawPRC(results, sp[0], "label propagation")
    curves.drawROC(results, sp, "MDCAP")
    #sp[0].legend(loc='upper right')
    #sp.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    #main()
    showCurves()


