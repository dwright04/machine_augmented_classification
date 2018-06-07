import sys
import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, roc_curve
from sklearn.cluster import KMeans
from sklearn import metrics

from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Dropout
from keras.initializers import Initializer
from keras.optimizers import SGD, Adadelta
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras.engine.topology import Layer
from keras import backend as K
from keras.models import load_model

sys.path.insert(0,'../../../../DEC-keras')
from DEC import DEC, ClusteringLayer, cluster_acc

sys.path.insert(0,'../../')
from dissolving_utils import get_cluster_centres, get_cluster_to_label_mapping
from dissolving_utils import pca_plot
from dissolving_utils import MappingLayer, MapInitializer
from dissolving_utils import load_dec

lcolours = ['#D6FF79', '#B0FF92', '#A09BE7', '#5F00BA', '#56CBF9', \
            '#F3C969', '#ED254E', '#CAA8F5', '#D9F0FF', '#46351D']

# DEC constants from DEC paper
batch_size = 256
lr         = 0.01
momentum   = 0.9
tol        = 0.001
maxiter    = 2e5
update_interval = 140 #perhaps this should be 1 for multitask learning
n_clusters = 10 # number of clusters to use
n_classes  = 2  # number of classes

class MultitaskDEC(DEC):
 def clustering(self, x, y=None, validation_data=None, tol=1e-3, update_interval=140, maxiter=2e4, save_dir='./results/dec', pretrained_weights=None):
    print('Update interval', update_interval)
    save_interval = x.shape[0] / self.batch_size * 5  # 5 epochs
    print('Save interval', save_interval)

    try:
      self.load_weights(pretrained_weights)
    except AttributeError:
      # initialize cluster centers using k-means
      print('Initializing cluster centers with k-means.')
      kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
      y_pred = kmeans.fit_predict(self.encoder.predict(x))
      y_pred_last = y_pred
      self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    y_p = self.predict_clusters(x)

    cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
      get_cluster_to_label_mapping_safe(y[:,1], y_p, n_classes, n_clusters)
    
    print(np.argmax((1-np.array(majority_class_fractions))*np.array(n_assigned_list)))
    cluster_to_label_mapping[np.argmax((1-np.array(majority_class_fractions))*np.array(n_assigned_list))] = 1
    
    a = Input(shape=(400,)) # input layer
    q_out = self.model(a)
    #d_out = Dropout(0.3)(q_out)
    pred = MappingLayer(cluster_to_label_mapping, output_dim=n_classes, \
      name='mapping', kernel_initializer=MapInitializer(cluster_to_label_mapping, n_classes))(q_out)
    self.model = Model(inputs=a, outputs=[pred, q_out])
    optimizer = SGD(lr=1e-1)
    #optimizer = 'adam'
    self.model.compile(optimizer=optimizer, loss={'mapping': 'categorical_crossentropy', 'model_3': 'kld'}, \
                                      loss_weights={'mapping': 1, 'model_3': 0.01})
    #import csv, os
    #if not os.path.exists(save_dir):
    #  os.makedirs(save_dir)

    #logfile = open(save_dir + '/dec_log.csv', 'w')
    #logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L'])
    #logwriter.writeheader()

    loss = [0, 0, 0]
    index = 0
    q = self.model.predict(x, verbose=0)[1]
    y_pred_last = q.argmax(1)
    best_val_loss = [np.inf, np.inf, np.inf]
    for ite in range(int(maxiter)):
      if ite % update_interval == 0:
        q = self.model.predict(x, verbose=0)[1]
        p = self.target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = y_pred
        y_pred = self.model.predict(x)[0]
        if y is not None:
          #acc = np.round(cluster_acc(y, y_pred), 5)
          #nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
          #ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
          loss = np.round(loss, 5)
          valid_p = self.target_distribution(self.model.predict(validation_data[0], verbose=0)[1])
          val_loss = np.round(self.model.test_on_batch(validation_data[0], [validation_data[1], valid_p]), 5)
          f, _, _, _ = one_percent_fpr(y[:,1], y_pred[:,1], 0.1)
          f = np.round(f, 5)
          f1 = np.round(f1_score(y[:,1], np.argmax(y_pred, axis=1)), 5)
          y_pred_valid = self.model.predict(validation_data[0])[0]
          f_valid, _, _, _ = one_percent_fpr(validation_data[1][:,1], y_pred_valid[:,1], 0.1)
          f_valid = np.round(f_valid, 5)
          f1_valid = np.round(f1_score(validation_data[1][:,1], np.argmax(y_pred_valid, axis=1)), 5)
          #logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss)
          #logwriter.writerow(logdict)
          #print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)
          print('Iter', ite, ' :1% fpr', f, ', F1=', f1, '; loss=', loss, \
                '; valid_loss=,', val_loss, '; valid 1% fpr=,', f_valid, ', valid F1=', f1_valid)
          if val_loss[1] < best_val_loss[1]: # only interested in classification improvements
            print('saving model: ', best_val_loss, ' -> ', val_loss)
            self.model.save_weights('best_val_loss.hf')
            best_val_loss = val_loss
      
        # check stop criterion
        """
        if ite > 0 and delta_label < tol:
          print('delta_label ', delta_label, '< tol ', tol)
          print('Reached tolerance threshold. Stopping training.')
          logfile.close()
          break
        """
        # train on batch
        if (index + 1) * self.batch_size > x.shape[0]:
          loss = self.model.train_on_batch(x=x[index * self.batch_size::],
                                           y=[y[index * self.batch_size::], \
                                              p[index * self.batch_size::]])
          index = 0
        else:
          loss = self.model.train_on_batch(x=x[index * self.batch_size:(index + 1) * self.batch_size],
                                           y=[y[index * self.batch_size:(index + 1) * self.batch_size], \
                                              p[index * self.batch_size:(index + 1) * self.batch_size]])
          index += 1

        # save intermediate model
        if ite % save_interval == 0:
        # save IDEC model checkpoints
          print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
          self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')

        ite += 1

    # save the trained model
    #logfile.close()
    print('saving model to:', save_dir + '/DEC_model_final.h5')
    self.model.save_weights(save_dir + '/DEC_model_final.h5')

    return y_pred

def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

def custom_loss(y_true, y_pred):
   #print(labelled_limit)
   #print(K.shape(y_true), K.shape(y_pred))
   kld_loss = kullback_leibler_divergence(y_true, y_pred)
   #print(K.shape(kld_loss))
   #print(K.shape(y_true[:labelled_limit]), K.shape(y_pred[:labelled_limit]))
   cc_loss = K.categorical_crossentropy(y_true[:labelled_limit], y_pred[:labelled_limit])
   
   print(K.shape(cc_loss), K.shape(kld_loss))
   
   return  cc_loss + 0.01 * kld_loss

def calc_fixed_fpr(y_true, y_preds, cluster_to_label_mapping, fom=0.1):
  #print(y_preds[:,np.where(np.array(cluster_to_label_mapping)==1)[0]].shape)
  #print(np.sum(y_preds[:,np.where(np.array(cluster_to_label_mapping)==1)[0]], axis=1).shape)
  #print(y_true.shape)
  return one_percent_fpr(y_true, np.sum(y_preds[:,np.where(np.array(cluster_to_label_mapping)==1)[0]], axis=1), fom)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), \
                            K.epsilon()))

class MultitaskLabelsByClusterDEC(DEC):

 def cluster_assignment(self, x, y, y_p, cluster_centres, cluster_to_label_mapping):
    x_encoded = self.encoder.predict(x)
    x_encoded_tiled = np.tile(x_encoded[:,:,np.newaxis], (1,1,self.n_clusters))
    cluster_centres_tiled = np.tile(cluster_centres[np.newaxis,:,:], (x.shape[0],1,1))
    euclidean_distances = np.squeeze(K.eval(euclidean_distance((x_encoded_tiled, cluster_centres_tiled))))
    cluster_preds = y_p
    y_c = []
    for i in range(x.shape[0]):
      l = np.argmax(y[i])
      c = cluster_preds[i]
      cl = cluster_to_label_mapping[c]
      if l == cl:
        y_c.append(c)
      else:
        ed = euclidean_distances[i][[np.where(cluster_to_label_mapping == l)]]
        ac = int(np.array(cluster_to_label_mapping)[np.where(cluster_to_label_mapping == l)][np.argmin(ed)])
        y_c.append(ac)
 
    # one hot encode these cluster assignements
    y_c = np_utils.to_categorical(y_c, self.n_clusters)
    return y_c
    
 def clustering(self, x_l, y_l, x_u, y_u, tol=1e-3, update_interval=140, maxiter=2e4, save_dir='./results/dec', pretrained_weights=None):
 
    # concatenate all the training vectors
    x = np.concatenate((x_l, x_u))
 
    print(x_l.shape, x_u.shape, x.shape)
    print('Update interval', update_interval)
    save_interval = x.shape[0] / self.batch_size * 5  # 5 epochs
    print('Save interval', save_interval)

    try:
      self.load_weights(pretrained_weights)
    except AttributeError:
      # initialize cluster centers using k-means
      print('Initializing cluster centers with k-means.')
      kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
      y_pred = kmeans.fit_predict(self.encoder.predict(x))
      y_pred_last = y_pred
      self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    # using only the labelled data set calculate the cluster label assignments
    y_pred = self.predict_clusters(x_l)
    global labelled_limit
    labelled_limit = x_l.shape[0]
    self.n_classes = len(np.unique(y_l))
    cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
      get_cluster_to_label_mapping_safe(y_l[:,1], y_pred, self.n_classes, self.n_clusters)
    
    # ensure that the cluster with the most real detections is assigned the real class
    print(np.argmax((1-np.array(majority_class_fractions))*np.array(n_assigned_list)))
    cluster_to_label_mapping[np.argmax((1-np.array(majority_class_fractions))*np.array(n_assigned_list))] = 1
    cluster_centres = np.squeeze(np.array(self.model.get_layer(name='clustering').get_weights()))
    
    # build training set based on euclidean distances
    """
    x_encoded = self.encoder.predict(x_l)
    x_encoded_tiled = np.tile(x_encoded[:,:,np.newaxis], (1,1,self.n_clusters))
    cluster_centres_tiled = np.tile(cluster_centres[np.newaxis,:,:], (x_l.shape[0],1,1))
    euclidean_distances = np.squeeze(K.eval(euclidean_distance((x_encoded_tiled, cluster_centres_tiled))))
    cluster_preds = y_p
    y_c = []
    for i in range(x_l.shape[0]):
      l = np.argmax(y_l[i])
      c = cluster_preds[i]
      cl = cluster_to_label_mapping[c]
      if l == cl:
        y_c.append(c)
      else:
        ed = euclidean_distances[i][[np.where(cluster_to_label_mapping == l)]]
        ac = int(np.array(cluster_to_label_mapping)[np.where(cluster_to_label_mapping == l)][np.argmin(ed)])
        y_c.append(ac)
 
    # one hot encode these cluster assignements
    y_c = np_utils.to_categorical(y_c, self.n_clusters)
    """
    # build the model
    a = Input(shape=(400,)) # input layer
    pred = self.model(a)
    self.mmodel = Model(inputs=a, outputs=pred)
    #optimizer = SGD(lr=1e-1)
    optimizer = 'adam'
    self.mmodel.compile(optimizer=optimizer, loss=custom_loss)

    # get soft assignments for every instance
    q = self.model.predict(x, verbose=0)
    # get the coresponding hard assignments
    y_pred_last = q.argmax(1)
    
    #loss = [np.inf, np.inf, np.inf]
    loss = np.inf
    #best_u_loss = [np.inf, np.inf, np.inf]
    best_u_loss = np.inf
    index = 0
    for ite in range(int(maxiter)):
      if ite % update_interval == 0:
        q = self.model.predict(x, verbose=0)
        p = self.target_distribution(q)  # update the auxiliary target distribution p
        # build training set based on euclidean distances
        y_c = self.cluster_assignment(x_l, y_l, y_pred, \
          np.squeeze(np.array(self.model.get_layer(name='clustering').get_weights())), \
          cluster_to_label_mapping)
        # replace labelled p wth labels
        p[:y_l.shape[0]] = y_c
        # evaluate the clustering performance
        y_pred = q.argmax(1)

        loss = np.round(loss, 5)
        y_u_c = self.cluster_assignment(x_u, y_u, self.predict_clusters(x_u), \
          np.squeeze(np.array(self.model.get_layer(name='clustering').get_weights())), \
          cluster_to_label_mapping)
        #print(x_u.shape, y_u_c.shape)
        labelled_limit = None
        u_loss = np.round(self.mmodel.evaluate(x_u, y_u_c), 5)
        #print(x_u.shape, y_u_c.shape)
        #print(q[:x_l.shape[0]].shape)
        #print(y_l.shape)
        f, _, _, _ = calc_fixed_fpr(y_l.argmax(1), q[:x_l.shape[0]], cluster_to_label_mapping, fom=0.1)
        f = np.round(f, 5)
        u_f, _, _, _ = calc_fixed_fpr(y_u.argmax(1), q[x_l.shape[0]:], cluster_to_label_mapping, fom=0.1)
        u_f = np.round(u_f, 5)
        print('Iter', ite, ' :1% fpr', f, '; loss=', loss, \
             '; u_loss=,', u_loss, '; unlabelled 1% fpr=,', u_f)
        if u_loss < best_u_loss: # only interested in classification improvements
          print('saving model: ', best_u_loss, ' -> ', u_loss)
          self.mmodel.save_weights('best_u_loss.hf')
          best_u_loss = u_loss
      
        # train on batch
        if (index + 1) * self.batch_size > x.shape[0]:
          loss = self.mmodel.train_on_batch(x=x[index * self.batch_size::],
                                           y=p[index * self.batch_size::])
          index = 0
        else:
          loss = self.mmodel.train_on_batch(x=x[index * self.batch_size:(index + 1) * self.batch_size],
                                           y=p[index * self.batch_size:(index + 1) * self.batch_size])
          index += 1
        labelled_limit = x_l.shape[0]
        # save intermediate model
        if ite % save_interval == 0:
        # save IDEC model checkpoints
          print('saving model to:', save_dir + '/DEC_model_multitask_' + str(ite) + '.h5')
          self.mmodel.save_weights(save_dir + '/DEC_model_multitask_' + str(ite) + '.h5')

        ite += 1

    # save the trained model
    print('saving model to:', save_dir + '/DEC_model_multitask_final.h5')
    self.mmodel.save_weights(save_dir + '/DEC_model_multitask_final.h5')

    return y_pred

def one_percent_fpr(y, pred, fom):
    fpr, tpr, thresholds = roc_curve(y, pred)
    FoM = 1-tpr[np.where(fpr<=fom)[0][-1]] # MDR at 1% FPR
    threshold = thresholds[np.where(fpr<=fom)[0][-1]]
    return FoM, threshold, fpr, tpr

def calc_f1_score(y_true, predicted_clusters, cluster_to_label_mapping):
  y_pred = []
  for i in range(len(y_true)):
    y_pred.append(cluster_to_label_mapping[predicted_clusters[i]])
  return f1_score(y_true, np.array(y_pred))

def get_cluster_to_label_mapping_safe(y, y_pred, n_classes, n_clusters):
  """Enusre at least one cluster assigned to each label.
  """
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
  #print(cluster_to_label_mapping)

  print(np.unique(y), np.unique(cluster_to_label_mapping))
  try:
    # make sure there is at least 1 cluster representing each class
    assert np.all(np.unique(y) == np.unique(cluster_to_label_mapping))
  except AssertionError:
    # if there is no cluster for a class then we will assign a cluster to that
    # class
    
    # find which class it is
    # ASSUMPTION - this task is binary
    
    diff = list(set(np.unique(y)) - set(np.unique(cluster_to_label_mapping)))[0]
      # we choose the cluster that contains the most examples of the class with no cluster
      
    one_hot = np_utils.to_categorical(y_pred[np.where(y==diff)[0]], \
                                        len(cluster_to_label_mapping))
                                      
    cluster_to_label_mapping[np.argmax(np.sum(one_hot, axis=0))] = int(diff)
  print(cluster_to_label_mapping)
  return cluster_to_label_mapping, n_assigned_list, majority_class_fractions

def load_data(sample_frac):
  """
    !!!!! NEED TO CHANGE THIS TO ENSURE NO OVERLAP BETWEEN OBJECTS FROM SAME NIGHT IN
    VALIDATION AND TRAINING SETS !!!!!
  """
  # load snhunters data set
  data = sio.loadmat('../../../../data/3pi_20x20_skew2_signPreserveNorm.mat')
  #x = np.concatenate((data['X'], data['testX']))
  #y = np.squeeze(np.concatenate((data['y'], data['testy'])))
  x = data['X']
  y = data['y']
  # split the data into training, validation and test sets similar to MNIST
  m = data['X'].shape[0]
  m = m - int(.25*m)
  split = int(sample_frac*m)
  print(m,split)
  x_train = data['X'][:split]
  y_train = np.squeeze(data['y'])[:split]
  x_valid = data['X'][split:]
  y_valid = np.squeeze(data['y'])[split:]
  x_test  = data['testX']
  y_test  = np.squeeze(data['testy'])

  return x, y, m, split, x_train, y_train, x_valid, y_valid, x_test, y_test

def multitask_volunteer_classifications_test(n):
  #_, _, _, _, x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(1)
  data = sio.loadmat('../../../../data/snhunters/zooniverse_test_set_detect_misaligned.mat')
  m = data['x'].shape[0]
  x_valid = np.nan_to_num(np.reshape(data['x'], (data['x'].shape[0], 400), order='F'))[:int(.5*m)]
  y_valid  = np.squeeze(data['y'])[:int(.5*m)]

  # load the pretrained DEC model for Supernova Hunters
  ae_weights  = '../../../../DEC-keras/results/snh/ae_weights_snh.h5'
  #ae_weights = './ae_weights_snh_multitask.h5'
  #dec_weights = '../../../../DEC-keras/results/snh/%d/DEC_model_final.h5'%n_clusters
  #dec_weights = None
  #dec_weights = './results/dec/DEC_model_final.h5'
  dec_weights = '../../../../DEC-keras/results/snh/10/DEC_model_final.h5'
  #dec = load_dec(x_valid, ae_weights, dec_weights, n_clusters, batch_size, lr, momentum)

  #dec.clustering(x_valid, np_utils.to_categorical(y_valid), pretrained_weights=dec_weights)

  #exit()

  
  for i in range(1,n+1):
    data = sio.loadmat('../../../../data/snhunters/3pi_20x20_supernova_hunters_batch_%d_signPreserveNorm_detect_misaligned.mat'%(i))
    try:
      x_train = np.concatenate((x_train, np.nan_to_num(np.reshape(data['X'], \
        (data['X'].shape[0], 400), order='F'))))
      y_train = np.concatenate((y_train, np.squeeze(data['y'])))
    except UnboundLocalError:
      x_train = np.nan_to_num(np.reshape(data['X'], (data['X'].shape[0], 400), \
        order='F'))
      y_train = np.squeeze(data['y'])
  
  u, indices = np.unique(x_train, return_index=True, axis=0)
  x_train = x_train[indices]
  y_train = y_train[indices]

  dec = MultitaskLabelsByClusterDEC(dims=[x_valid.shape[-1], 500, 500, 2000, 10], \
    n_clusters=n_clusters, batch_size=batch_size)
  dec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
                     ae_weights=ae_weights,
                     x=x_train)
  dec.model.load_weights(dec_weights)
  '''
  cluster_centres = get_cluster_centres(dec)
  labels = [str(i) for i in range(n_clusters)]
  lcolours = ['#CAA8F5', '#D6FF79', '#A09BE7', '#5F00BA', '#56CBF9', \
              '#F3C969', '#ED254E', '#B0FF92', '#D9F0FF', '#46351D']
  pca_plot(dec.encoder, x_train, cluster_centres, y=y_train, labels=labels, lcolours=[lcolours[0], lcolours[1]])
  '''
  dec.clustering(x_train, np_utils.to_categorical(y_train), \
    x_valid, np_utils.to_categorical(y_valid), \
    pretrained_weights=dec_weights, maxiter=maxiter)

  #y_pred = dec.model.predict(x_valid)[0]
  #f, _, _, _ = one_percent_fpr(y_valid, y_pred[:,1], 0.1)
  #print(f)
  #print(f1_score(y_valid, np.argmax(y_pred, axis=1)))


  #cluster_centres = get_cluster_centres(dec)
  #labels = [str(i) for i in range(n_clusters)]
  #lcolours = ['#CAA8F5', '#D6FF79', '#A09BE7', '#5F00BA', '#56CBF9', \
  #            '#F3C969', '#ED254E', '#B0FF92', '#D9F0FF', '#46351D']
  #pca_plot(dec.encoder, x_train, cluster_centres, y=y_train, labels=labels, lcolours=[lcolours[0], lcolours[1]])

multitask_volunteer_classifications_test(3)

