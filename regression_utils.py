import numpy as np
import os
import scipy.io as sio
from sklearn.linear_model import LinearRegression as Linreg
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.svm import SVR

def GetModel(regname=None):
    if regname.lower()=='lasso':
        regressor = Lasso()
    if regname.lower()=='linear_model':
        regressor=Linreg()
    if regname.lower()=='svr':
        regressor= SVR()
    return regressor


def MakeRegression(model=[],X=[],y=[],inner_cv=None,outer_cv=None,feat_selection=False):
    if inner_cv==None:

        scores = cross_validate(model, X, y, cv=outer_cv,
                                scoring=('r2', 'neg_mean_squared_error'),
                                return_train_score=True)
    return scores
