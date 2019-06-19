#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#
# Description: causal association prediction


import os
import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import misim
import duallp
import matrics


def main():
    
    dataset = pd.read_csv("data/hmdd_causal.txt", sep="\t")
    outdir = "inputdata/hmdd"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    trainset = dataset
    testset = pd.DataFrame(columns=['mir', 'disease', 'causality'])
    trainset.to_csv("%s/trainset.txt" % outdir, sep="\t", index=False)
    testset.to_csv("%s/testset.txt" % outdir, sep="\t", index=False)
    matrics.saveMatrics(trainset, outdir)
    duallp.main(outdir)
    
    results = pd.read_csv("%s/dlp_results.txt" % outdir, sep="\t")
    results.to_csv("results/hmdd_results.txt", sep="\t", index=False)

if __name__ == "__main__":

    main()
