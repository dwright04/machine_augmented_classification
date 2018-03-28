import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda
from keras.initializers import Initializer
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.engine.topology import Layer
from keras import backend as K

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), \
                            K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def get_pairs_auto(dec, x, y, cluster_centres, cluster_to_label_mapping, \
                   majority_class_fractions, n_clusters):

  cluster_preds = dec.predict_clusters(x)
  
  m = x.shape[0]
  
  im = []
  cc = []
  labels = []
  print(np.all(np.unique(y) == np.unique(cluster_to_label_mapping)))
  try:
    # make sure there is at least 1 cluster representing each class
    assert np.all(np.unique(y) == np.unique(cluster_to_label_mapping))
  except AssertionError:
    # if there is no cluster for a class then we will assign a cluster to that
    # class
    
    # find the which class it is
    # ASSUMPTION - this task is binary
    try:
      diff = list(set(np.unique(y)) - set(np.unique(cluster_to_label_mapping)))[0]
    
    
      # we choose the cluster that contains the most examples of the class with no cluster
      one_hot = np_utils.to_categorical(cluster_preds[np.where(y==diff)], \
                                        len(cluster_to_label_mapping))
                                      
      cluster_to_label_mapping[np.argmax(np.sum(one_hot, axis=0))] = diff
    except IndexError:
      pass
  
  for i in range(m):
    l = y[i]
    c = cluster_preds[i]
    cl = cluster_to_label_mapping[c]
    if l == cl:
      # if subject label == the label of its assigned cluster
      im += [x[i]]
      cc += [cluster_centres[c]]
      labels += [1]
      for j in range(n_clusters):
        if j != c:
          im += [x[i]]
          cc += [cluster_centres[c]]
          labels += [0]
  
    elif l != cl:
      # if the subject != the label of its assigned cluster
      # and there exists a cluster with the same label as this subject
      ed = []
      encodedx = dec.encoder.predict(x[i][np.newaxis])
      #print(encodedx.shape, np.array(cluster_centres).shape)
      #print(np.tile(encodedx, (len(cluster_centres),1)).T.shape)
      #print(encodedx - np.tile(encodedx, (len(cluster_centres),1)).T[:,0])
      ed = K.eval(euclidean_distance((encodedx, \
        np.array(cluster_centres[np.where(cluster_to_label_mapping == l)]))))
      #print(K.eval(euclidean_distance((encodedx, np.array(cluster_centres)))))
      ac = int(np.array(cluster_to_label_mapping)[np.where(cluster_to_label_mapping == l)][np.argmin(ed)])
      im += [x[i]]
      try:
        cc += [cluster_centres[ac]]
      except IndexError:
        print(ac, type(ac))
        exit()
      labels += [1]
      for j in range(n_clusters):
        if j != ac:
          im += [x[i]]
          cc += [cluster_centres[ac]]
          labels += [0]

  im = np.array(im)
  cc = np.array(cc)
  labels = np.array(labels)
  order = np.random.permutation(im.shape[0])
  im = im[order]
  cc = cc[order]
  labels = labels[order]
  return im, cc, labels, cluster_to_label_mapping

def get_pairs_auto_with_noise(dec, x, y, cluster_centres, cluster_to_label_mapping, \
                   majority_class_fractions, n_clusters, noise=0.1):

  cluster_preds = dec.predict_clusters(x)
  
  m = x.shape[0]
  
  im = []
  cc = []
  labels = []
  
  try:
    # make sure there is at least 1 cluster representing each class
    assert np.all(np.unique(y) == np.unique(cluster_to_label_mapping))
  except AssertionError:
    # if there is no cluster for a class then we will assign a cluster to that
    # class
    
    # find the which class it is
    # ASSUMPTION - this task is binary
    diff = list(set(np.unique(y)) - set(np.unique(cluster_to_label_mapping)))[0]
  
    # we choose the cluster that contains the most examples of the class with no cluster
    one_hot = np_utils.to_categorical(cluster_preds[np.where(y==diff)], \
                                      len(cluster_to_label_mapping))
                                      
    cluster_to_label_mapping[np.argmax(np.sum(one_hot, axis=0))] = diff

  choices = np.unique(y)
  for i in range(m):
    l = y[i]
    if np.random.random() < noise:
      l = np.random.choice(choices)
      #print(y[i], l)
    c = cluster_preds[i]
    cl = cluster_to_label_mapping[c]
    if l == cl:
      # if subject label == the label of its assigned cluster
      im += [x[i]]
      cc += [cluster_centres[c]]
      labels += [1]
    elif l != cl:
      # if the subject != the label of its assigned cluster
      # and there exists a cluster with the same label as this subject
      ed = []
      encodedx = dec.encoder.predict(x[i][np.newaxis])
      ed = K.eval(euclidean_distance((encodedx, \
        np.array(cluster_centres[np.where(cluster_to_label_mapping == l)]))))
      im += [x[i]]
      cc += [cluster_centres[np.array(cluster_to_label_mapping)[np.where(cluster_to_label_mapping == l)][np.argmin(ed)]]]
      labels += [1]
  im = np.array(im)
  cc = np.array(cc)
  labels = np.array(labels)
  return im, cc, labels, cluster_to_label_mapping

def train_siamese(dec, centres, im, cl, labels, epochs=50, \
                  split_frac=0.75, callbacks=[]):
    
  base_network = Model(dec.model.input, \
    dec.model.get_layer('encoder_%d' % (dec.n_stacks - 1)).output)
  embedded_dim = centres.shape[1]
  i = Input(shape=(embedded_dim,))
  o = Lambda(lambda x: 1*x)(i)
  m = Model(inputs=i, outputs=o)

  input_dim = im.shape[1]
  input_a = Input(shape=(input_dim,))
  input_b = Input(shape=(embedded_dim,))
  processed_a = base_network(input_a)
  processed_b = m(input_b)
  distance = Lambda(euclidean_distance,
                    output_shape=eucl_dist_output_shape)([processed_a, processed_b])
  sigmoid = Dense(1, activation='sigmoid')(distance)
  model = Model([input_a, input_b], sigmoid)
  model.compile(loss='binary_crossentropy', optimizer='adam')
  limit = int(split_frac*im.shape[0])
  model.fit([im[:limit], cl[:limit]], labels[:limit], \
    validation_data=([im[limit:], cl[limit:]], labels[limit:]), \
    batch_size=256, epochs=epochs, callbacks=callbacks)
  return model, base_network

def train_siamese_online(dec, x, centres, im, cl, labels, epochs=50, \
                        split_frac=0.75, callbacks=[]):
    
  base_network = Model(dec.model.input, \
    dec.model.get_layer('encoder_%d' % (dec.n_stacks - 1)).output)
  embedded_dim = centres.shape[1]
  i = Input(shape=(embedded_dim,))
  o = Lambda(lambda x: 1*x)(i)
  m = Model(inputs=i, outputs=o)
  input_dim = x.shape[1]
  input_a = Input(shape=(input_dim,))
  input_b = Input(shape=(embedded_dim,))
  processed_a = base_network(input_a)
  processed_b = m(input_b)
  distance = Lambda(euclidean_distance, \
    output_shape=eucl_dist_output_shape)([processed_a, processed_b])
  sigmoid = Dense(1, activation='sigmoid')(distance)
  model = Model([input_a, input_b], sigmoid)
  model.compile(loss='binary_crossentropy', optimizer='adam')
  limit = int(split_frac*im.shape[0])
  for i in range(epochs):
    for j in range(limit):
      order = np.random.permutation(limit)
      d1 = im[:limit][order]
      d2 = cl[:limit][order]
      print(np.array(labels).shape)
      ls = np.array(labels)[:limit][order]
      model.fit([d1[i][np.newaxis], d2[i][np.newaxis]], ls[i][np.newaxis], \
        validation_data=([im[limit:], cl[limit:]], labels[limit:]), \
        batch_size=1, epochs=1, callbacks=callbacks)
  return model, base_network
