import numpy as np
import scipy.io as sio
import os

main_path=os.getcwd()
data_path=os.path.join(main_path,'Results','predict_PSD')
result_file='Res_100ROI_{bd}_{reg}.mat'

bande='Beta'
regressor='lasso'

Res=sio.loadmat(os.path.join(data_path,result_file.format(bd=bande,reg=regressor)))['results_ROI']
print(Res[0,1])
