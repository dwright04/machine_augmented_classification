import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda
from keras.initializers import Initializer
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils
from keras.engine.topology import Layer
from keras import backend as K

from sklearn.metrics.pairwise import euclidean_distances

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
  #print(np.all(np.unique(y) == np.unique(cluster_to_label_mapping)))
  #print(cluster_to_label_mapping)
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
  #print(cluster_to_label_mapping)
  for i in range(m):
    l = y[i]
    c = cluster_preds[i]
    cl = cluster_to_label_mapping[c]
    if l == cl:
      # if subject label == the label of its assigned cluster
      im += [x[i]]
      cc += [cluster_centres[c]]
      labels += [1]
      #for j in range(n_clusters):
      #  if j != c:
      #    im += [x[i]]
      #    cc += [cluster_centres[c]]
      #    labels += [0]
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
      im += [x[i]]
      cc += [cluster_centres[c]]
      labels += [0]
      #for j in range(n_clusters):
      #  if j != ac:
      #    im += [x[i]]
      #    cc += [cluster_centres[ac]]
      #

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

def get_pairs_triplet_selection(dec, x, y, cluster_predictions, cluster_to_label_mapping):
  """
    Assumes one classification per training example. NO DUPLICATES.
    
    Assumes this batch of data (i.e number of training examples) is small 
    enough such that we can calculate the m-way euclidean distances.
  """
  x_embedded = dec.encoder.predict(x)
  #x_embedded = np.tile(x_embedded[:,:,np.newaxis], (1,1,dec.n_clusters))
  
  distances = euclidean_distances(x_embedded, x_embedded)
  # squared distance for triplet loss from Facenet Schroff et al. 2015
  distances *= distances
  print(distances.shape)
  
  m = x.shape[0]
  
  pairs = []
  labels = []

  for i in range(m):
    l = y[i]
    c = cluster_predictions[i]
    cl = cluster_to_label_mapping[c]
    pos_eds = distances[i,y==1]
    neg_eds = distances[i,y!=l]
    for j in range(m):
      if j < i:
        continue
      # if we already haven't considered this index
      if y[j] == l and cluster_predictions[j] == c:
        # if the comparison subject label has the same label and is assigned
        # to the same cluster then create a positve pair
        pairs += [[x[i], x[j]]]
        labels += [1]
        pos_ed = distances[i,j]
        try:
          # create the semi-hard negative pair based on this positive pair: Facenet Schroff et al. 2015
          neg_index = np.where(distances[i] == np.min(distances[i][distances[i] > pos_ed]))[0][0]
          #print(neg_index)
          #print(pos_ed, distances[i][neg_index])
          pairs += [[x[i], x[neg_index]]]
          labels += [0]
        except ValueError:
          continue
      elif y[j] == l and cluster_predictions[j] != c:
        # otherwise if the comparison subject has the same label, but is
        # assigned to a different cluster then do not create the positive pair
        continue

  return np.array(pairs), np.array(labels)

def train_siamese_triplet_selection(dec, pairs, labels, optimizer, epochs=10, callbacks=[], split_frac=.75):

  base_network = Model(dec.model.input, \
    dec.model.get_layer('encoder_%d' % (dec.n_stacks - 1)).output)

  input_dim = pairs.shape[-1]
  input_a = Input(shape=(input_dim,))
  input_b = Input(shape=(input_dim,))
  processed_a = base_network(input_a)
  processed_b = base_network(input_b)

  distance = Lambda(euclidean_distance,
                    output_shape=eucl_dist_output_shape)([processed_a, processed_b])
  sigmoid = Dense(1, activation='sigmoid')(distance)
  model = Model([input_a, input_b], sigmoid)

  model.compile(loss='binary_crossentropy', optimizer=optimizer)

  val_split = int(split_frac*pairs.shape[0])
  model.fit([pairs[:val_split, 0], pairs[:val_split, 1]], labels[:val_split], \
    validation_data=([pairs[val_split:, 0], pairs[val_split:, 1]], labels[val_split:]), \
    batch_size=256, epochs=epochs, callbacks=callbacks)
  return model, val_split

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
  #sigmoid = Dense(2, activation='sigmoid')(distance)
  model = Model([input_a, input_b], sigmoid)
  #optimizer = Adadelta(lr=0.01)
  optimizer = SGD(lr=0.001)
  #model.compile(loss='binary_crossentropy', optimizer='adam')
  #model.compile(loss='binary_crossentropy', optimizer=optimizer)
  #model.compile(loss='categorical_crossentropy', optimizer=optimizer)
  #model.compile(loss='binary_crossentropy', optimizer=optimizer)
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
