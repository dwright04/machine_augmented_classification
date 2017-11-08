import sys
import numpy as np
import scipy.io as sio
from time import time
sys.path.insert(0,'../DEC-keras/')
from DEC import DEC, ClusteringLayer, cluster_acc

def load_data(dataFile):
  data = sio.loadmat(dataFile)
  x = np.concatenate((data['X'], data['testX']))
  y = np.concatenate((data['y'], data['testy']))
  return x, y

def train_dec(x, y, n_clusters, save_dir):
  batch_size = 256
  lr         = 0.01
  momentum   = 0.9
  tol        = 0.001
  maxiter         = 3e4
  update_interval = 1e3
  
  dec = DEC(dims=[x.shape[-1], 500, 500, 2000, 10],
            n_clusters=n_clusters,
            batch_size=batch_size)

  dec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
                       ae_weights='../DEC-keras/ae_weights_snh.h5', x=x)

  try:
    dec_snh.load_weights(save_dir+'/DEC_model_final.h5')
    y_pred = dec_snh.predict_clusters(x)
  except IOError:
    t0 = time()
    y_pred = dec_snh.clustering(x, y=y, tol=tol, maxiter=maxiter,
                                update_interval=update_interval, save_dir=save_dir)
    print('clustering time: ', (time() - t0))
  print('acc:', cluster_acc(y, y_pred))

def main():
  save_dir = '../DEC-keras/results/dec/3pi_20x20_skew2_signPreserveNorm'

  x, y = load_data('../../../data/3pi_20x20_skew2_signPreserveNorm.mat')
  for i in [10, 100, 1000]:
    train_dec(x, y, i, save_dir+'/%d'%(i))

  save_dir = '../DEC-keras/results/dec/3pi_20x20_skew2_zeroOneScaling'

  x, y = load_data('../../../data/3pi_20x20_skew2_zeroOneScaling.mat')
  for i in [10, 100, 1000]:
    train_dec(x, y, i, save_dir+'/%d'%(i))

if __name__ == '__main__':
  main()
