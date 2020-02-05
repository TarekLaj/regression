import numpy as np
import scipy.io as sio
import os
import pandas as pd
from regression_utils import MakeRegression,GetModel
from sklearn.model_selection import StratifiedKFold
####Load PSD and csv files ####
data_path=os.getcwd()
Psd_file=os.path.join(data_path,'PSD_all_bands.mat')
PSD_all_bands=sio.loadmat(Psd_file)['psd']
csv_file=os.path.join(data_path,'data_N45_all.csv')
df=pd.read_csv(csv_file,sep=';')
target_name='Treatment'
target=df[target_name].tolist()


bandes=['Delta','Theta','Alpha','Beta','Gamma1','Gamma2','Gamma3','GammaL']
regressor='Lasso'
inner_cv=None
outer_cv=10
feature_selection=False
for bdi,bd in enumerate(bandes):
    X=np.squeeze(PSD_all_bands[:,bdi,:]).T
    model=GetModel(regressor)
    regsc=MakeRegression(model=model,X=X,y=target,inner_cv=None,outer_cv=None,feat_selection=False)
    print(regsc)
