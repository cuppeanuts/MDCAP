#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#
# Draw Curves

import matplotlib.pyplot as sp
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
from sklearn.utils.fixes import signature


def drawPRC(result, sp, methodlabel):
	''' Accept a result dataframe whose 'cond' coulmn is binary series
		representing the ture condition and 'pred' column is the normalized
		score of predicton.
	'''
	print("drawing prc curve...")
	sp.set_xlabel('Recall')
	sp.set_ylabel('Precision')
	sp.set_ylim([0.0, 1.0])
	sp.set_xlim([0.0, 1.0])
	sp.set_title('2-class Precision-Recall curve')
	
	precision, recall, threshold = precision_recall_curve(result['cond'], result['pred'])
	average_precision = average_precision_score(result['cond'], result['pred'])
	myauc = auc(recall, precision)
	step_kwargs = ({'step': 'post'} 
		if 'step' in signature(sp.fill_between).parameters
		else {})
	sp.step(recall, precision, alpha=0.2, where='post')
	sp.fill_between(recall, precision, alpha=0.2, label=methodlabel+" (AUC = %.3f)"%myauc, **step_kwargs)
	


	return myauc

def drawROC(result, sp, methodlabel):
	print("drawing roc curve...")
	sp.set_xlabel('FPR')
	sp.set_ylabel('TPR')
	sp.set_ylim([0.0, 1.0])
	sp.set_xlim([0.0, 1.0])
	#sp.set_title('2-class ROC curve')

	fpr, tpr, thresholds = roc_curve(result['cond'], result['pred'])
	myauc = auc(fpr, tpr)
	step_kwargs = ({'step': 'post'} 
		if 'step' in signature(sp.fill_between).parameters
		else {})
	sp.step(fpr, tpr, color="green", alpha=1, where='post')
        print("AUC: %.3f"%myauc)
	#sp.fill_between(fpr, tpr, color="green", alpha=1, label=methodlabel+" (AUC = %.3f)"%myauc, **step_kwargs)


	return myauc
