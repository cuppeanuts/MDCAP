#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#
# MISIM Model

import re
import random
import math
import networkx as nx
import pandas as pd
import numpy as np


def loadMtree(mtreefile):
    ''' Accept a mtree file and return a dict whose keys are disease
        names and values are the list of mids. 
        (ex. 'Heart Diseases': ['C14.280'])
    '''
    with open(mtreefile, 'r') as fi:
        f = [row.strip().split(";") for row in fi.readlines()]
    mtree = dict([])
    for row in f:
        name = row[0]
        mid = row[1]
        if name not in mtree:
            mtree[name] = [mid]
        else:
            mtree[name].append(mid)
    return mtree


def genDAG(tlist):
    ''' Accept a mid list and return a DAG of the disease.
    '''
    nodes = set([])
    edges = []
    selfid = ";".join(tlist)
    for tree in tlist:
        nl = tree.split(".")
        for i in range(len(nl)):
            if i == 0:
                parent = nl[i]
                nodes.add(parent)
            else:
                child = ".".join([parent, nl[i]])
                if child in tlist:
                    child = selfid
                nodes.add(child)
                edges.append((parent, child))
                parent = child
    G = nx.DiGraph()
    G.add_nodes_from(list(nodes))
    G.add_edges_from(edges)
    return G


def dVal(node, d, DAG):
    ''' Calculate the sematic Contribute(D value) of a node(disease) in DAG 
        to the disease on root. d is semantic contribution factor. 
        Accept a DAG, a node in DAG and a factor d and return the dVal of the 
        node.
    '''
    if DAG.successors(node) == []:
        return 1.0
    else:
        return max([d*dVal(child, d, DAG) for child in DAG.successors(node)])


def semVal(DAG, d):
    ''' Calculate the sematic value of the DAG root which is define as the sum
        of the d value of all nodes in DAG.
        Accept a DAG and return a sematic value.
    '''
    return sum([dVal(n, d, DAG) for n in DAG.nodes()])


def semSim(G1, G2, d):
    ''' Calculate the sematic similarity of two DAG, which is define as the 
        ratio of the sum of the d values of all overlap nodes in two DAG and 
        the sum of two DAG sematic values. 
        Accept two DAG and return the similarity.
    '''
    overlap_nodes = list(set(G1.nodes()) & set(G2.nodes()))
    sv1 = semVal(G1, d)
    sv2 = semVal(G2, d)
    olsv = sum([dVal(n, d, G1) + dVal(n, d, G2) for n in overlap_nodes])
    similarity = olsv/(sv1 +sv2)
    return similarity


def misim(dl1, dl2, mtree, d=0.5):
    ''' Construction of miRNA functional simialriy matrix.
        Accept two miRNA associated disease lists, a mtree dict and 
        a contribute factor and return the similariy of two miRNA.
    '''
    
    all_disease = list(set(dl1 + dl2))
    dag = {d:genDAG(mtree[d]) for d in all_disease}

    simall = {}
    for i in dl1:
        simall[i] = {}
        for j in dl2:
            if j in simall and i in simall[j]:
                simall[i][j] = simall[j][i]
                continue
            simall[i][j] = semSim(dag[i], dag[j], d)

    S1 = sum([max([simall[i][j] for j in dl2]) for i in dl1])
    S2 = sum([max([simall[i][j] for i in dl1]) for j in dl2])
    msim = (S1 + S2)/float(len(dl1) + len(dl2))

    return msim

