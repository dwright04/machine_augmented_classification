import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn import metrics
from matplotlib import rcParams
from time import time

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda
from keras.initializers import Initializer
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.engine.topology import Layer
from keras import backend as K

np.random.seed(0)

sys.path.insert(0,'../../DEC-keras/')
from DEC import DEC, ClusteringLayer, cluster_acc

def get_cluster_to_label_mapping(y, y_pred, n_classes, n_clusters):

  one_hot_encoded = np_utils.to_categorical(y, n_classes)

  cluster_to_label_mapping = []
  n_assigned_list = []
  majority_class_fractions = []
  majority_class_pred = np.zeros(y.shape)
  for cluster in range(n_clusters):
    cluster_indices = np.where(y_pred == cluster)[0]
    n_assigned_examples = cluster_indices.shape[0]
    n_assigned_list.append(n_assigned_examples)
    cluster_labels = one_hot_encoded[cluster_indices]
    cluster_label_fractions = np.mean(cluster_labels, axis=0)
    majority_cluster_class = np.argmax(cluster_label_fractions)
    cluster_to_label_mapping.append(majority_cluster_class)
    majority_class_pred[cluster_indices] += majority_cluster_class
    majority_class_fractions.append(cluster_label_fractions[majority_cluster_class])
    print(cluster, n_assigned_examples, majority_cluster_class, cluster_label_fractions[majority_cluster_class])
  print(cluster_to_label_mapping)
  return cluster_to_label_mapping, n_assigned_list, majority_class_fractions

class MapInitializer(Initializer):
    
  def __init__(self, mapping, n_classes):
    self.mapping = mapping
    self.n_classes = n_classes

  def __call__(self, shape, dtype=None):
    return K.one_hot(self.mapping, self.n_classes)
    #return K.ones(shape=(100,10))

  def get_config(self):
    return {'mapping': self.mapping, 'n_classes': self.n_classes}

class MappingLayer(Layer):

  def __init__(self, mapping, output_dim, kernel_initializer, **kwargs):
  #def __init__(self, mapping, output_dim, **kwargs):
    self.output_dim = output_dim
    # mapping is a list where the index corresponds to a cluster and the value is the label.
    # e.g. say mapping[0] = 5, then a label of 5 has been assigned to cluster 0
    self.n_classes = np.unique(mapping).shape[0]      # get the number of classes
    self.mapping = K.variable(mapping, dtype='int32')
    self.kernel_initializer = kernel_initializer
    super(MappingLayer, self).__init__(**kwargs)

  def build(self, input_shape):
  
    self.kernel = self.add_weight(name='kernel', 
                                  shape=(input_shape[1], self.output_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=False)
  
    super(MappingLayer, self).build(input_shape)  # Be sure to call this somewhere!

  def call(self, x):
    return K.softmax(K.dot(x, self.kernel))

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_dim)

def build_model(dec, x, cluster_to_label_mapping, input_shape, ae_weights, \
                save_dir, lr, momentum, n_classes=10, metrics = ['acc']):
  a = Input(shape=(input_shape,)) # input layer
  q = dec.model(a)
  pred = MappingLayer(cluster_to_label_mapping, output_dim=n_classes, \
    kernel_initializer=MapInitializer(cluster_to_label_mapping, n_classes))(q)
  model = Model(inputs=a, outputs=pred)
  model.compile(loss='categorical_crossentropy', optimizer='adam', \
    metrics=metrics)
  return model

def get_cluster_anchors(x, y, dec, cluster_to_label_mapping, n_clusters):
  m_all = np.shape(x)[0]
  cluster_centres = \
    np.squeeze(np.array(dec.model.get_layer(name='clustering').get_weights()))
    
  x_embedded = dec.extract_feature(x)
  anchors = []
  anchor_indices = []
  for i in range(n_clusters):
    indices_assigned = np.where(y==cluster_to_label_mapping[i])
    m = indices_assigned[0].shape[0]
    c = np.tile(cluster_centres[i][np.newaxis], (m,1))
    c_all = np.tile(cluster_centres[i][np.newaxis], (m_all,1))
    distances_assigned = \
      np.linalg.norm(x_embedded[indices_assigned] - c, axis=1)
    
    distances = np.linalg.norm(x_embedded - c_all, axis=1)
    anchor_indices.append(np.argmin(distances))
    anchors.append(x[np.argmin(distances)])
  return np.array(anchors), np.array(anchor_indices)

def get_cluster_centres(dec):
  return np.squeeze(np.array(dec.model.get_layer(name='clustering').get_weights()))

def calculateAccuracy(y, y_pred):
  return 100*np.sum(y_pred == y) / len(y_pred)

def precent_fpr(y_true, y_pred, percent=0.01):
  fpr, tpr, thresholds = roc_curve(y_true, y_pred)
  FoM = 1-tpr[np.where(fpr<=percent)[0][-1]] # MDR at 1% FPR
  return FoM

def pca_plot(base_network, x, cluster_centres, y=None, labels=[], \
             lcolours=[], ulcolour='#747777', ccolour='#4D6CFA'):
    
  pca = PCA(n_components=2)
  x_pca = pca.fit_transform(base_network.predict(x))
  c_pca = pca.transform(cluster_centres)
  fig = plt.figure(figsize=(6,6))
  ax = fig.add_subplot(111)
  if np.any(y):
    unique_targets = list(np.unique(y))
    if -1 in unique_targets:
      ax.scatter(x_pca[np.where(y==-1),0], x_pca[np.where(y==-1),1], marker='o', s=20, \
        color=ulcolour, alpha=0.1)
      unique_targets.remove(-1)
    for l in unique_targets:
        ax.scatter(x_pca[np.where(y==l),0], x_pca[np.where(y==l),1], \
          marker='o', s=20, color=lcolours[l], alpha=0.6, label=labels[l])
  else:
    ax.scatter(x_pca[:,0], x_pca[:,1], marker='o', s=20, \
      color=ulcolour, alpha=0.1)
  ax.scatter(c_pca[:,0], c_pca[:,1], marker='o', s=40, color=ccolour, \
    alpha=1.0, label='cluster centre')

  for i in range(len(cluster_centres)):
    ax.text(c_pca[i,0], c_pca[i,1], str(i), size=20)
  plt.axis('off')
  plt.legend(ncol=3)
  plt.show()

class FrameDumpCallback(keras.callbacks.Callback):
  def __init__(self, base_network, x, cluster_centres, file_path, y=None, labels=[], \
               lcolours=[], ulcolour='#747777', ccolour='#4D6CFA', \
               epoch_incrementer=0):
    self.base_network = base_network
    self.x = x
    self.cluster_centres = cluster_centres
    self.y = y
    self.file_path = file_path
    self.epoch_incrementer = epoch_incrementer
  
    self.labels = labels
    self.lcolours = lcolours
    self.ulcolour = ulcolour
    self.ccolour = ccolour

  def on_batch_end(self, batch, logs):
    encoder = self.base_network
    self.epoch_incrementer += 1
  
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(encoder.predict(self.x))
    c_pca = pca.transform(self.cluster_centres)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    if np.any(self.y):
      unique_targets = list(np.unique(self.y))
      if -1 in unique_targets:
        ax.scatter(x_pca[np.where(self.y==-1),0], \
          x_pca[np.where(self.y==-1),1], marker='o', s=20, \
          color=self.ulcolour, alpha=0.1)
      
        unique_targets.remove(-1)
      for l in unique_targets:
        ax.scatter(x_pca[np.where(self.y==l),0], \
                   x_pca[np.where(self.y==l),1], \
                   marker='o', s=20, color=self.lcolours[l], \
                   alpha=0.6, label=self.labels[l])
    else:
      ax.scatter(x_pca[:,0], x_pca[:,1], marker='o', s=20, \
                 color=self.ulcolour, alpha=0.1)
    ax.scatter(c_pca[:,0], c_pca[:,1], marker='o', s=40, color=self.ccolour, \
               alpha=1.0, label='cluster centre')

    for i in range(len(self.cluster_centres)):
      ax.text(c_pca[i,0], c_pca[i,1], str(i), size=20)
    plt.axis('off')
    plt.legend(ncol=3)
    plt.savefig(self.file_path+'/frame_%06d.png'%(self.epoch_incrementer))
