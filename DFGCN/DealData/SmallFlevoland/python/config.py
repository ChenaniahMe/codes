#the number of cpu
import scipy.io as sio
CORE=20
corrA=sio.loadmat("label.mat")['label'].shape[0]
corrB=sio.loadmat("label.mat")['label'].shape[1]
adj_name = "adj"