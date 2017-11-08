import sys
import numpy as np
import scipy.io as sio
from time import time

from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
sys.path.insert(0,'../../DEC-keras/')
from DEC import DEC, ClusteringLayer, cluster_acc
np.random.seed(0)

def custom_kld_with_penalty(y_true, y_pred):
  y_true = K.clip(y_true, K.epsilon(), 1)
  y_pred = K.clip(y_pred, K.epsilon(), 1)
  kld = K.sum(y_true * K.log(y_true / y_pred), axis=-1)
  
  ro = K.cast(K.tile([K.shape(y_true)[1] / K.shape(y_true)[0]], [K.shape(y_true)[1]]), 'float64')
  #ro = K.cast(K.tile([K.cast(1, 'int32') / K.shape(y_true)[0]], [K.shape(y_true)[1]]), 'float64')
  #ro = K.cast(K.tile([K.cast(1, 'int32') / 1e4], [K.shape(y_true)[1]]), 'float64')
  #ro = K.cast(K.tile([0.01], [K.shape(y_true)[1]]), 'float64')
  ro_j = K.cast(K.sum(y_true, axis=0) / K.tile([K.cast(K.shape(y_true)[0], 'float32')], [K.shape(y_true)[1]]), 'float64')
    
  penalty = K.mean((ro * K.log(ro / ro_j)) + (1. - ro) * K.log((1. - ro) / (1. - ro_j)), axis=-1)
  return kld + K.cast(penalty, 'float32')

def cost(y_true, y_pred):
  y_true = K.clip(y_true, K.epsilon(), 1)
  y_pred = K.clip(y_pred, K.epsilon(), 1)
  kld = K.sum(y_true * K.log(y_true / y_pred), axis=-1)
  print(K.argmax(y_pred, axis=1))
  regTerm = K.sum(K.one_hot(K.argmax(y_pred, axis=1), K.shape(y_true)[1]), axis=0)
  regTerm = K.sum(K.cast(regTerm == 0,'float32'))
  return kld + regTerm

# might be worth considering calculation of p and set a minimum value of 1 subject per cluster.
data = sio.loadmat('../../data/3pi_20x20_skew2_signPreserveNorm.mat')
#data = sio.loadmat('../../data/3pi_20x20_skew2_zeroOneScaling.mat')
x = np.concatenate((data['X'], data['testX']))
y = np.squeeze(np.concatenate((data['y'], data['testy'])))

#save_dir = '../DEC-keras/results/testing/snh/100' # testing directory for custom loss DEC
save_dir = '.'

dec_snh100 = DEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=100, batch_size=256)
dec_snh100.initialize_model(optimizer=SGD(lr=0.01, momentum=0.9),
                            ae_weights='../../DEC-keras/results/snh/ae_weights_snh.h5',
                            x=x, loss='kld')
try:
  dec_snh100.load_weights(save_dir+'/DEC_model_final.h5')
  y_pred = dec_snh100.predict_clusters(x)
except IOError:
  t0 = time()
  y_pred = dec_snh100.clustering(x, y=y, tol=0.001, maxiter=2e4,
                                 update_interval=140, save_dir=save_dir)
  print('clustering time: ', (time() - t0))
print('acc:', cluster_acc(y, y_pred))
"""
q = dec_snh100.model.predict(x, verbose=0)
p = dec_snh100.target_distribution(q)
print(cost(K.variable(p), K.variable(q)))
print(K.eval(cost(K.variable(p), K.variable(q))))
print(K.eval(cost(K.variable(p), K.variable(q))).shape)
"""
n_classes = 2 # We know this from the data, but may not for a real project

one_hot_encoded = np_utils.to_categorical(y, n_classes)

cluster_to_label_mapping = [] # will be a mapping from the cluster index to the label assigned to that cluster
majority_class_pred = np.zeros(y.shape)
for cluster in range(100):
  cluster_indices = np.where(y_pred == cluster)[0]
  n_assigned_examples = cluster_indices.shape[0]
  cluster_labels = one_hot_encoded[cluster_indices]
  cluster_label_fractions = np.mean(cluster_labels, axis=0)
  majority_cluster_class = np.argmax(cluster_label_fractions)
  cluster_to_label_mapping.append(majority_cluster_class)
  majority_class_pred[cluster_indices] += majority_cluster_class
  print(cluster, n_assigned_examples, majority_cluster_class, cluster_label_fractions[majority_cluster_class])
print(cluster_to_label_mapping)
