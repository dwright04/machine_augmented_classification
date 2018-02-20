import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn import metrics
from matplotlib import rcParams
from time import time

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda
from keras.initializers import Initializer
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.engine.topology import Layer
from keras import backend as K

np.random.seed(0)

sys.path.insert(0,'../../../DEC-keras/')
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

def build_model(dec, x, cluster_to_label_mapping, lr, momentum, n_classes=10, input_shape=784,
                ae_weights='../../../DEC-keras/results/mnist/ae_weights.h5', save_dir='../../../DEC-keras/results/mnist/10/',
                metrics = ['acc']):
  #dec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
  #                     ae_weights=ae_weights,
  #                     x=x)
  #dec.load_weights(save_dir+'/DEC_model_final.h5')
  a = Input(shape=(input_shape,)) # input layer
  q = dec.model(a)
  pred = MappingLayer(cluster_to_label_mapping, output_dim=n_classes, kernel_initializer=MapInitializer(cluster_to_label_mapping, n_classes))(q)
  model = Model(inputs=a, outputs=pred)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
  return model

def get_cluster_anchor(x, y, dec, cluster_to_label_mapping, n_clusters):
  m_all = np.shape(x)[0]
  cluster_centres = np.squeeze(np.array(dec.model.get_layer(name='clustering').get_weights()))
  x_embedded = dec.extract_feature(x)
  anchors = []
  anchor_indices = []
  for i in range(n_clusters):
    indices_assigned = np.where(y==cluster_to_label_mapping[i])
    m = indices_assigned[0].shape[0]
    c = np.tile(cluster_centres[i][np.newaxis], (m,1))
    c_all = np.tile(cluster_centres[i][np.newaxis], (m_all,1))
    distances_assigned = np.linalg.norm(x_embedded[indices_assigned] - c, axis=1)
    distances = np.linalg.norm(x_embedded - c_all, axis=1)
    anchor_indices.append(np.argmin(distances))
    anchors.append(x[np.argmin(distances)])
  return np.array(anchors), np.array(anchor_indices)

def calculateAccuracy(y, y_pred):
  return 100*np.sum(y_pred == y) / len(y_pred)

def precent_fpr(y_true, y_pred, percent=0.01):
  fpr, tpr, thresholds = roc_curve(y_true, y_pred)
  FoM = 1-tpr[np.where(fpr<=percent)[0][-1]] # MDR at 1% FPR
  return FoM

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def load_snhunters_data():
  data = sio.loadmat('../../../data/3pi_20x20_skew2_signPreserveNorm.mat')
  x = np.concatenate((data['X'], data['testX']))
  y = np.squeeze(np.concatenate((data['y'], data['testy'])))
  x_train = data['X']
  x_test  = data['testX']
  y_train = np.squeeze(data['y'])
  y_test  = np.squeeze(data['testy'])
  return x, y, x_train, y_train, x_test, y_test

def load_snhunter_DEC(x, n_clusters, batch_size, momentum):
  dec_snh = DEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, batch_size=batch_size)
  dec_snh.initialize_model(optimizer=SGD(lr=0.01, momentum=momentum),
                            ae_weights='../../../DEC-keras/results/snh/%d/ae_weights_snh.h5'%n_clusters,
                            x=x, loss='kld')
  save_dir = '../../../DEC-keras/results/snh/%d'%n_clusters
  try:
    dec_snh.load_weights(save_dir+'/DEC_model_final.h5')
    y_pred = dec_snh.predict_clusters(x)
  except IOError:
    t0 = time()
    y_pred = dec_snh.clustering(x, y=y, tol=tol, maxiter=maxiter,
                                update_interval=update_interval, save_dir=save_dir)
    print('clustering time: ', (time() - t0))
  dec_snh.model.summary()
  cluster_centres = dec_snh.model.get_layer(name='clustering').get_weights()[0]
  return dec_snh, cluster_centres

def get_pairs_hardcoded(dec_snh, x, y, anchors, n_clusters):
  cluster_preds = dec_snh.predict_clusters(x)
  im = []
  cl = []
  labels = []
  for cluster in range(n_clusters):
    xs = x[np.where(cluster_preds == cluster)]
    ys = y[np.where(cluster_preds == cluster)]
    for i in range(len(xs)):
      if ys[i] == 0 and cluster != 5:
        im += [xs[i]]
        cl += [anchors[cluster]]
        labels += [1]
      if ys[i] == 1:
        im += [xs[i]]
        cl += [anchors[5]]
        labels += [1]
        
  im = np.array(im)
  cl = np.array(cl)
  labels = np.array(labels)
  return im, cl, labels

def get_pairs_auto(dec, x, y, anchors, cluster_to_label_mapping, majority_class_fractions, n_clusters):

  cluster_preds = dec.predict_clusters(x)
  
  m = x.shape[0]
  
  im = []
  an = []
  labels = []
  
  try:
    # make sure there is at least 1 cluster representing each class
    assert np.all(np.unique(y) == np.unique(cluster_to_label_mapping))
  except AssertionError:
    # if there is no cluster for a class then we will assign a cluster to that class
    
    # find the which class it is
    # ASSUMPTION - this task is binary
    diff = list(set(np.unique(y)) - set(np.unique(cluster_to_label_mapping)))[0]
  
    # we choose the cluster that contains the most examples of the class with no cluster
    cluster_to_label_mapping[np.argmax(np.sum(np_utils.to_categorical(cluster_preds[np.where(y==diff)], len(cluster_to_label_mapping)), axis=0))] = diff

  for i in range(m):
    #print(i)
    l = y[i]
    c = cluster_preds[i]
    cl = cluster_to_label_mapping[c]
    pu = majority_class_fractions[c]
    if l == cl:
      # if subject label == the label of its assigned cluster
      im += [x[i]]
      an += [anchors[c]]
      labels += [1]
    elif l != cl and l in cluster_to_label_mapping:
      # if the subject != the label of its assigned cluster
      # and there exists a cluster with the same label as this subject
      ed = []
      encodedx = dec.encoder.predict(x[i][np.newaxis])
      #print(encodedx.shape, np.array(anchors).shape)
      #print(np.tile(encodedx, (len(anchors),1)).T.shape)
      #print(encodedx - np.tile(encodedx, (len(anchors),1)).T[:,0])
      
      #for j in range(len(anchors[cluster_to_label_mapping==l])):
        # for all the cluster anchors with the same label as this subject
        # find the closest in the euclidean sense.
      #  ed.append(K.eval(euclidean_distance((encodedx, anchors[j]))))
      ed = K.eval(euclidean_distance((encodedx, np.array(anchors))))
      im += [x[i]]
      an += [anchors[np.argmin(ed)]]
      labels += [1]
      
  im = np.array(im)
  an = np.array(an)
  labels = np.array(labels)
  return im, an, labels

def pca_plot(base_network, x, anchors, y=None):
  pca = PCA(n_components=2)
  x_pca = pca.fit_transform(base_network.predict(x))
  a_pca = pca.transform(anchors)
  fig = plt.figure(figsize=(6,6))
  ax = fig.add_subplot(111)
  if np.any(y):
    ax.scatter(x_pca[np.where(y==0),0], x_pca[np.where(y==0),1], marker='o', s=20, color='#747777', alpha=0.6, label='bogus')
    ax.scatter(x_pca[np.where(y==1),0], x_pca[np.where(y==1),1], marker='o', s=20, color='#DA3E52', alpha=0.6, label='real')
  else:
    ax.scatter(x_pca[:,0], x_pca[:,1], marker='o', s=20, color='#747777', alpha=0.1)
  ax.scatter(a_pca[:,0], a_pca[:,1], marker='o', s=20, color='#68C3D4', alpha=1.0, label='anchor')
  for i in range(len(anchors)):
    ax.text(a_pca[i,0], a_pca[i,1], str(i))
  plt.axis('off')
  plt.legend()
  plt.show()

def train_siamese(dec_snh, x, anchors, im, cl, labels, epochs=50):
  base_network = Model(dec_snh.model.input, dec_snh.model.get_layer('encoder_%d' % (dec_snh.n_stacks - 1)).output)
  embedded_dim = anchors.shape[1]
  i = Input(shape=(embedded_dim,))
  o = Lambda(lambda x: 1*x)(i)
  m = Model(inputs=i, outputs=o)
  input_dim = x.shape[1]
  input_a = Input(shape=(input_dim,))
  input_b = Input(shape=(embedded_dim,))
  processed_a = base_network(input_a)
  processed_b = m(input_b)
  distance = Lambda(euclidean_distance,
                    output_shape=eucl_dist_output_shape)([processed_a, processed_b])
  sigmoid = Dense(1, activation='sigmoid')(distance)
  model = Model([input_a, input_b], sigmoid)
  #sgd = SGD(lr=1e-3)
  #model.compile(loss='binary_crossentropy', optimizer=sgd)
  model.compile(loss='binary_crossentropy', optimizer='adam')
  model.fit([im, cl], labels, batch_size=256, epochs=epochs)
  return model, base_network

def reclustering(x, y, x_train, y_train, x_test, y_test, anchors, base_network, n_clusters, batch_size, momentum):

  redec_snh = DEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, batch_size=batch_size)
  redec_snh.initialize_model(optimizer=SGD(lr=0.01, momentum=momentum),
                            ae_weights='../../../../DEC-keras/results/snh/%d/ae_weights_snh.h5'%n_clusters,
                            x=x, loss='kld')
  save_dir = '../../../DEC-keras/results/snh/%d'%n_clusters
  try:
    redec_snh.load_weights(save_dir+'/DEC_model_final.h5')
    y_pred = redec_snh.predict_clusters(x_train)
  except IOError:
    t0 = time()
    y_pred = redec_snh.clustering(x, y=y, tol=tol, maxiter=maxiter,
                                update_interval=update_interval, save_dir=save_dir)
    print('clustering time: ', (time() - t0))

  for i,l in enumerate(base_network.layers):
    w = l.get_weights()
    redec_snh.encoder.layers[i].set_weights(w)

  # prepare DEC model
  redec_snh.model.compile(loss='kld', optimizer=SGD(lr=0.01, momentum=0.9))
  redec_snh.model.get_layer(name='clustering').set_weights([anchors])
  redec_snh.model.summary()

  y_pred = redec_snh.predict_clusters(x_train)

  cluster_to_label_mapping, n_assigned_list, majority_class_fractions_orig = \
  get_cluster_to_label_mapping(y_train, y_pred, 2, n_clusters=n_clusters)

  get_cluster_to_label_mapping(y_test, redec_snh.predict_clusters(x_test), 2, n_clusters=n_clusters)

  return redec_snh

def siamese_test():
  batch_size = 256
  lr         = 0.01 # learning rate
  momentum   = 0.9
  tol        = 0.001 # tolerance - if clustering stops if less than this fraction of the data changes cluster on an interation
  maxiter         = 2e4
  update_interval = 140
  n_clusters = 10
  
  x, y, x_train, y_train, x_test, y_test = load_snhunters_data()
  
  dec_snh, cluster_centres = load_snhunter_DEC(x, n_clusters, batch_size, momentum)

  y_pred = dec_snh.predict_clusters(x_train)

  cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
  get_cluster_to_label_mapping(y_train, y_pred, 2, n_clusters=n_clusters)
  
  get_cluster_to_label_mapping(y_test, dec_snh.predict_clusters(x_test), 2, n_clusters=n_clusters)
  
  anchors, anchor_indices = get_cluster_anchor(x_train, y_train, dec_snh, cluster_to_label_mapping, n_clusters)
  anchors = dec_snh.extract_feature(anchors)

  #im, cl, labels = get_pairs_hardcoded(dec_snh, x_train, y_train, anchors, n_clusters)
  im, cl, labels = get_pairs_auto(dec_snh, x, y, anchors, cluster_to_label_mapping, majority_class_fractions, n_clusters)
  
  model, base_network = train_siamese(dec_snh, x_train, anchors, im, cl, labels, epochs=20)

  pca_plot(base_network, x, anchors, y=y_train)

  pca_plot(base_network, x_test, anchors, y=y_test)

  dec = reclustering(x, y, x_train, y_train, x_test, y_test, anchors, base_network, n_clusters, batch_size, momentum)

  cluster_to_label_mapping, n_assigned_list, majority_class_fractions_orig = \
  get_cluster_to_label_mapping(y_train, dec.predict_clusters(x_train), 2, n_clusters=n_clusters)

  model = build_model(dec, x_train, cluster_to_label_mapping, lr, momentum, n_classes=2, input_shape=400,
              ae_weights='../../../DEC-keras/results/snh/10/ae_weights_snh.h5',
              save_dir='../../../DEC-keras/results/snh/10/', metrics = ['acc'])

  limit = int(.75*x_train.shape[0])
  model.fit(x_train[:limit], np_utils.to_categorical(y_train[:limit]),
            validation_data=(x_train[limit:], np_utils.to_categorical(y_train[limit:])),
            batch_size=batch_size, epochs=100)

  pca_plot(dec.encoder, x, anchors, y=y_train)

  pca_plot(dec.encoder, x_test, anchors, y=y_test)

  get_cluster_to_label_mapping(y_train, dec.predict_clusters(x_train), 2, n_clusters=n_clusters)

  get_cluster_to_label_mapping(y_test, dec.predict_clusters(x_test), 2, n_clusters=n_clusters)

def sub_clustering_test():
  batch_size = 256
  lr         = 0.01 # learning rate
  momentum   = 0.9
  tol        = 0.001 # tolerance - if clustering stops if less than this fraction of the data changes cluster on an interation
  maxiter         = 2e4
  update_interval = 140
  n_clusters = 10
  
  x, y, x_train, y_train, x_test, y_test = load_snhunters_data()

  dec, cluster_centres = load_snhunter_DEC(x, n_clusters, batch_size, momentum)

  y_pred = dec.predict_clusters(x)

  cluster_to_label_mapping, n_assigned_list, majority_class_fractions_orig = \
  get_cluster_to_label_mapping(y, y_pred, 2, n_clusters=n_clusters)

  anchors, anchor_indices = get_cluster_anchor(x, y, dec, cluster_to_label_mapping, n_clusters)

  anchors = dec.extract_feature(anchors)
  
  pca_plot(dec.encoder, x, anchors, y=y)

  y_pred = dec.predict_clusters(x_train)
  xs = x_train[np.where(y_pred == 5)]
  ys = y_train[np.where(y_pred == 5)]
  print(xs.shape)

  dec1 = DEC(dims=[xs.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, batch_size=batch_size)
  dec1.initialize_model(optimizer=SGD(lr=0.01, momentum=momentum),
                            ae_weights='../../../DEC-keras/results/snh/%d/ae_weights_snh.h5'%n_clusters,
                            x=xs, loss='kld')
  save_dir = '../../../DEC-keras/results/snh/%d/cluster5'%n_clusters
  try:
    dec1.load_weights(save_dir+'/DEC_model_final.h5')
    y_pred = dec1.predict_clusters(xs)
  except IOError:
    t0 = time()
    y_pred = dec1.clustering(xs, y=ys, tol=tol, maxiter=maxiter,
                                update_interval=update_interval, save_dir=save_dir)
    print('clustering time: ', (time() - t0))
  dec1.model.summary()

  y_pred = dec1.predict_clusters(xs)

  cluster_to_label_mapping, n_assigned_list, majority_class_fractions_orig = \
  get_cluster_to_label_mapping(ys, y_pred, 2, n_clusters=n_clusters)

  anchors, anchor_indices = get_cluster_anchor(xs, ys, dec1, cluster_to_label_mapping, n_clusters)

  anchors = dec1.extract_feature(anchors)
  
  pca_plot(dec1.encoder, xs, anchors, y=ys)

  y_pred = dec.predict_clusters(x_test)
  xs_test = x_test[np.where(y_pred == 5)]
  ys_test = y_test[np.where(y_pred == 5)]

  pca_plot(dec1.encoder, xs_test, anchors, y=ys_test)

  get_cluster_to_label_mapping(ys_test, dec1.predict_clusters(xs_test), 2, n_clusters=n_clusters)

  model = build_model(dec1, xs, cluster_to_label_mapping, lr, momentum, n_classes=2, input_shape=400,
                ae_weights='../../../DEC-keras/results/snh/ae_weights.h5', save_dir='../../../DEC-keras/results/snh/10/cluster5',
                metrics = ['acc'])

  limit = int(.75*xs.shape[0])
  model.fit(xs[:limit], np_utils.to_categorical(ys[:limit]),
            validation_data=(xs[limit:], np_utils.to_categorical(ys[limit:])),
            batch_size=batch_size, epochs=50)

  pca_plot(dec1.encoder, xs_test, anchors, y=ys_test)

def main():
  siamese_test()
  #sub_clustering_test()

if __name__ == '__main__':
  main()
