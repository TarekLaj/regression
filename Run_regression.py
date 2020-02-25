import numpy as np
import scipy.io as sio
import os
import pandas as pd
from regression_utils import MakeRegression,GetModel
from sklearn.model_selection import StratifiedKFold

######################## Load PSD and csv files ################################

data_path=os.getcwd()
Psd_file=os.path.join(data_path,'PSD_all_bands.mat')
PSD_all_bands=sio.loadmat(Psd_file)['psd']
csv_file=os.path.join(data_path,'data_N45_all.csv')
df=pd.read_csv(csv_file,sep=';')

########################### Test On/off  #######################################

test=True
if test:
    bandes=['Delta']
else:
    bandes=['Delta','Theta','Alpha','Beta','Gamma1','Gamma2','Gamma3','GammaL']

####################### Set regression parameters ##############################

target_name='Radio'# Set the target name
target=df[target_name].tolist()

regressor='MLP'  # Regressor name
inner_cv=5         # Inner cross validation used to optimise the model's params
outer_cv=10        # Outer cv used to train and test data
optimise=True           # Turn to True if you want to optimise the model
FeatSelect=False   # Set to True if you want to run feature selection
target_name='Radio'
target=df[target_name].tolist()

########################### Run Regression #####################################
for bdi,bd in enumerate(bandes):
    print('Runing regression for bande {}'.format(bd))
    X=np.squeeze(PSD_all_bands[:,bdi,:]).T
    model=GetModel(regname=regressor,
                   optimisation=optimise,
                   cv=inner_cv)
    results=MakeRegression(model=model,
                        X=X,
                        y=target,
                        inner_cv=inner_cv,
                        outer_cv=outer_cv,
                        feat_selection=FeatSelect)
    print(results)
