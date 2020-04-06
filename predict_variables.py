import numpy as np
import scipy.io as sio
import os
import pandas as pd
from regression_utils import MakeRegression,GetModel
from sklearn.model_selection import StratifiedKFold

######################## Load PSD and csv files ################################

data_path=os.getcwd()
save_path=os.path.join(data_path,'Results')
if not os.path.exists(save_path):
    os.makedirs(save_path)
Psd_file=os.path.join(data_path,'PSD_100_ROIs.mat')
PSD_all_bands=sio.loadmat(Psd_file)['PSD']
var_file=os.path.join(data_path,'data_N45_cortico_ITMTX.xls')
df=pd.read_excel(var_file)

########################### Test On/off  #######################################

test=True
if test:
    bandes=['Delta']
else:
    bandes=['Delta','Theta','Alpha','Beta','Gamma1','Gamma2','Gamma3','GammaL']

####################### Set regression parameters ##############################

target_name='ITMTX'# Set the target name : 'ITMTX', 'Cortico'
target=df[target_name].tolist()
regressor='Lasso'  # Regressor name
inner_cv=10      # Inner cross validation used to optimise the model's params
outer_cv=5      # Outer cv used to train and test data
optimise=True    # Turn to True if you want to optimise the model
FeatSelect=False # Set to True if you want to run feature selection
stat=True
nbperms=10
# [float(i) for i in target]
# print(type(target[0]))
########################### Run Regression #####################################
for bdi,bd in enumerate(bandes):
    print('Runing regression for bande {}'.format(bd))
    scores,perm_sc,pvals=[],[],[]

    X=np.squeeze(PSD_all_bands[:,bdi,:]).T
    model=GetModel(regname=regressor,
                   optimisation=optimise,
                   cv=inner_cv)
    score,permutation_score,pvalue=MakeRegression(model=model,
                                                X=X,
                                                y=target,
                                                inner_cv=inner_cv,
                                                outer_cv=outer_cv,
                                                stat=stat,
                                                nperms=nbperms,
                                                njobs=-1)
    scores.append(score)
    perm_sc.append(permutation_score)
    pvals.append(pvalue)
    print(score)
save_file=os.path.join(save_path,'predict_variables','Res_100ROI_{b}_{reg}'.format(b=bd,
                                                                    reg=regressor))
sio.savemat(save_file,{'scores':scores,'perm_sc':perm_sc,'pvals':pvals})
