#! /bin/env python

"""
Apply scikit-learn to sample data
"""

import os, sys

import numpy as np

from numpy import genfromtxt

import pandas as pd
#import pandas.io.data
from pandas_datareader import data, wb
from pandas import DataFrame


import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10 # increase plot size a bit...
rcParams['axes.formatter.useoffset'] = False  # disable scalar formatter / offset in axes labels

from sklearn import tree
from IPython.display import Image 
from sklearn.externals.six import StringIO  
import pydot 

    
def get_pdf_from_csv( my_file_csv ):
    """ Get a clean pandas dataframe from a csv file
    """
    pdf = pd.read_csv(my_file_csv)
    
    perf_pix= pdf[pdf['Mean_PQMas'] == 16383.0]  # where(Mean_PQMas=16383)
    
    # change non 'W_*' into NoWater
    perf_pix.ix[perf_pix.Class_name.str.match('^W_*')==False, 'Class_name'] = 'NoWater'

    # change W_* into Water; all other classes into NotWater. Binary classes
    perf_pix.ix[perf_pix.Class_name.str.match('^W_*'), 'Class_name'] = 'Water'
    
    # do some stats to inspect the samples
    # Group by Pixel Class_name, then count the number of rows in each group
    print ("perfect pixels classes:", perf_pix.groupby('Class_name').count())
    
    print ("Raw data classes:", pdf.groupby('Class_name').count())
    
    return perf_pix
    
    
def get_sample_target(pdf):
    """
    Prepare sample-target from a pandas dataframe pdf
    """
    #get the column names as list    
    clm_list = []
    for column in pdf.columns: clm_list.append(column)
    print (clm_list[7],clm_list[10],clm_list[12])
    
    #select the columns values into numpy array
    #X = pdf[clm_list[3:14]].values
    # select 3 most important features
    X= pdf[[clm_list[7],clm_list[10],clm_list[12]] ].values
    Y = pdf[clm_list[1]].values

    print X.shape, type(X)
    print Y.shape
    
    return (X, Y)
    

def get_model_by_train(X,Y):
    """
    apply machine learning alg to sample X and Y=target
    """
    
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf= clf.fit(X,Y)
    
    dot_filename="de_tree.dot"
    with open(dot_filename, 'w') as f: f = tree.export_graphviz(clf, out_file=f)

    # output the model params:
    # !dot -Tpdf de_tree.dot -o de_tree.pdf
    
    return  clf

def check_model(clf, X, Y):
    """ Self-check the model using function predict
    """
    n_sample=1000
    print ("quick check n_sample= ", n_sample)

    # Or get a random sample to compare
    
    maxsample=Y.shape[0]
    for ns in range(1,n_sample):    
        isample=np.random.random_integers(0,maxsample)
        p_Y=clf.predict(X[isample,:].reshape(1, -1) )
        if p_Y[0] != Y[isample]:
            print ("Wrong!!!  quick check at sample=", isample, p_Y, Y[isample])
            print  X[isample,:]  # should keepi X_orig Y_orig to see what it is
        else:
            pass
            #print("Correct Predict")

    return

def main(csvfile):
    pdf=get_pdf_from_csv(csvfile)
    (X,Y)=get_sample_target(pdf)
    mod = get_model_by_train(X,Y)
    check_model(mod, X, Y)


if __name__== '__main__':
    
    csvfile=sys.argv[1]
    
    main(csvfile)
