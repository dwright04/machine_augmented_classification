import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.utils import np_utils

sys.path.insert(0,'../../../../DEC-keras')
from DEC import DEC

sys.path.insert(0,'../../')
from dissolving_utils import get_cluster_centres, get_cluster_to_label_mapping
from dissolving_utils import pca_plot, FrameDumpCallback, percent_fpr

sys.path.insert(0,'../')
from siamese_utils import get_pairs_auto, get_pairs_auto_with_noise, train_siamese, train_siamese_online
from siamese_utils import get_pairs_triplet_selection, train_siamese_triplet_selection

def load_snhunters_dec(x, ae_weights, dec_weights, n_clusters, batch_size, lr, \
                   momentum):
  dec = DEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, batch_size=batch_size)
  ae_weights = ae_weights
  dec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
                       ae_weights=ae_weights,
                       x=x, loss='kld')
  dec.load_weights(dec_weights)
  dec.model.summary()
  return dec

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

def test():
  # constants
  batch_size = 256
  lr         = 0.01
  momentum   = 0.9
  tol        = 0.001
  maxiter         = 2e4
  update_interval = 140

  n_clusters = 10
  n_classes  = 2

  lcolours = ['#D6FF79', '#B0FF92', '#A09BE7', '#5F00BA', '#56CBF9', \
              '#F3C969', '#ED254E', '#CAA8F5', '#D9F0FF', '#46351D']
  labels = [str(i) for i in range(n_clusters)]
  
  ae_weights  = '../../../../DEC-keras/results/snh/ae_weights_snh.h5'
  dec_weights = '../../../../DEC-keras/results/snh/%d/DEC_model_final.h5'%n_clusters

  percent = 0.1 # for figure of merit
  # load snhunters data set
  data = sio.loadmat('../../../../data/3pi_20x20_skew2_signPreserveNorm.mat')
  x = np.concatenate((data['X'], data['testX']))
  y = np.squeeze(np.concatenate((data['y'], data['testy'])))
  # split the data into training, validation and test sets
  m = data['X'].shape[0]
  m = m - int(.25*m)
  sample_frac = 0.2
  split = int(sample_frac*m)
  print(split)
  x_train = data['X'][:split]
  y_train = np.squeeze(data['y'])[:split]
  x_valid = data['X'][split:]
  y_valid = np.squeeze(data['y'])[split:]
  x_test  = data['testX']
  y_test  = np.squeeze(data['testy'])

  # load pretrained DEC model
  dec = load_snhunters_dec(x, ae_weights, dec_weights, n_clusters, \
    batch_size, lr, momentum)

  # predict training set cluster assignments
  y_pred = dec.predict_clusters(x_train)

  # inspect the clustering and simulate volunteer labelling of random sample (the training set)
  cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
    get_cluster_to_label_mapping(y_train, y_pred, n_classes, n_clusters)
  yp = []
  for i in range(len(y_pred)):
    yp.append(cluster_to_label_mapping[y_pred[i]])
  print(percent_fpr(y_train, np.array(yp), percent))
  print(f1_score(y_train, np.array(yp)))

  y_valid_pred = dec.predict_clusters(x_valid)
  ypv = []
  for i in range(len(y_valid_pred)):
    ypv.append(cluster_to_label_mapping[y_valid_pred[i]])
  #print(ypv)
  print(percent_fpr(y_valid, np.array(ypv), percent))
  print(f1_score(y_valid, np.array(ypv)))
  
  # extract the cluster centres
  cluster_centres = get_cluster_centres(dec)

  # determine current unlabelled samples
  y_plot = np.array(y[:m],dtype='int')
  y_plot[split:] = -1

  # reduce embedding to 2D and plot labelled and unlabelled training set samples
  #pca_plot(dec.encoder, x[:m], cluster_centres, y=y_plot, labels=labels, \
  #           lcolours=lcolours)

  # get siamese training pairs
  im, cc, ls, cluster_to_label_mapping = \
    get_pairs_auto(dec, x_train, y_train, cluster_centres, \
      cluster_to_label_mapping, majority_class_fractions, n_clusters)

  #im, cc, ls, cluster_to_label_mapping = \
  #  get_pairs_auto_with_noise(dec, x_train, y_train, cluster_centres, \
  #    cluster_to_label_mapping, majority_class_fractions, n_clusters)
  """
  mcheckpointer = ModelCheckpoint(filepath='saved_models/weights.best..hdf5', \
                                  verbose=1, save_best_only=True)

  base_network = Model(dec.model.input, \
    dec.model.get_layer('encoder_%d' % (dec.n_stacks - 1)).output)
  fcheckpointer = FrameDumpCallback(base_network, x, cluster_centres, \
    './video', y=y_plot, labels=labels, lcolours=lcolours)
  """
  #callbacks = [mcheckpointer, fcheckpointer]
  callbacks = []

  model, base_network = train_siamese(dec, cluster_centres, im, cc, ls, \
    epochs=5, split_frac=0.75, callbacks=callbacks)
  #model, base_network = train_siamese_online(dec, x, cluster_centres, im, cc, ls, \
  #  epochs=1, split_frac=0.75, callbacks=[])

  y_pred = dec.predict_clusters(x_valid)
  
  cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
    get_cluster_to_label_mapping(y_valid, y_pred, n_classes, n_clusters)
  yp = []
  for i in range(len(y_pred)):
    yp.append(cluster_to_label_mapping[y_pred[i]])
  print(percent_fpr(y_valid, np.array(yp), percent))
  print(f1_score(y_valid, np.array(yp)))
  #pca_plot(dec.encoder, x_valid, cluster_centres, y=y_valid, labels=labels, \
  #           lcolours=lcolours)

  y_pred = dec.predict_clusters(x[:m])

  majority_class_fractions = np.array(majority_class_fractions)
  print(np.where(majority_class_fractions==np.nan))
  print(majority_class_fractions[np.where(majority_class_fractions==np.nan)])
  majority_class_fractions[np.isnan(majority_class_fractions)] = np.inf
  print(majority_class_fractions)
  print(np.argmin(majority_class_fractions))
  for j in range(1,5):
    #selection = np.where(y_pred[j*split:(j+1)*split] == np.argmin(majority_class_fractions))
    #x_train = np.concatenate((x_train, x[:m][j*split:(j+1)*split][selection]))
    #y_train = np.concatenate((y_train, y[:m][j*split:(j+1)*split][selection]))
    x_train = np.concatenate((x_train, x[:m][j*split:(j+1)*split]))
    y_train = np.concatenate((y_train, y[:m][j*split:(j+1)*split]))
  
    im, cc, ls, cluster_to_label_mapping = \
      get_pairs_auto(dec, x_train, y_train, cluster_centres, \
        cluster_to_label_mapping, majority_class_fractions, n_clusters)

    callbacks = []
  
    model, base_network = train_siamese(dec, cluster_centres, im, cc, ls, \
      epochs=1, split_frac=0.75, callbacks=callbacks)

  #x_train = x[:2*split]
  #y_train = y[:2*split]
  #y_pred = dec.predict_clusters(x_train)
  
  #cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
  #  get_cluster_to_label_mapping(y_train, y_pred, n_classes, n_clusters)

    y_pred = dec.predict_clusters(x_valid)
  
    cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
      get_cluster_to_label_mapping(y_valid, y_pred, n_classes, n_clusters)
    majority_class_fractions = np.array(majority_class_fractions)
    majority_class_fractions[np.where(majority_class_fractions==np.nan)] = np.inf
    yp = []
    for i in range(len(y_pred)):
      yp.append(cluster_to_label_mapping[y_pred[i]])
    print(percent_fpr(y_valid, np.array(yp), percent))
    print(f1_score(y_valid, np.array(yp)))
    #print(percent_fpr(y_valid, y_pred, percent))
    #pca_plot(dec.encoder, x_valid, cluster_centres, y=y_valid, labels=labels, \
    #          lcolours=lcolours)

  print('test set')
  y_pred = dec.predict_clusters(x_test)
  cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
    get_cluster_to_label_mapping(y_test, y_pred, n_classes, n_clusters)
  majority_class_fractions = np.array(majority_class_fractions)
  majority_class_fractions[np.where(majority_class_fractions==np.nan)] = np.inf
  yp = []
  for i in range(len(y_pred)):
    yp.append(cluster_to_label_mapping[y_pred[i]])
  print(percent_fpr(y_test, np.array(yp), percent))
  print(f1_score(y_test, np.array(yp)))

def volunteer_classification_test():
  # constants
  batch_size = 256
  lr         = 0.01
  momentum   = 0.9
  tol        = 0.001
  maxiter         = 2e4
  update_interval = 140

  n_clusters = 10
  n_classes  = 2

  lcolours = ['#D6FF79', '#B0FF92', '#A09BE7', '#5F00BA', '#56CBF9', \
              '#F3C969', '#ED254E', '#CAA8F5', '#D9F0FF', '#46351D']
              
  labels = [str(i) for i in range(n_clusters)]
  
  ae_weights  = '../../../../DEC-keras/results/snh/ae_weights_snh.h5'
  dec_weights = '../../../../DEC-keras/results/snh/%d/DEC_model_final.h5'%n_clusters

  percent = 0.1 # for figure of merit

  # load snhunters data set
  data = sio.loadmat('../../../../data/snhunters/zooniverse_test_set.mat')
  x_valid = np.nan_to_num(np.reshape(data['x'], (data['x'].shape[0], 400), order='F'))
  y_valid  = np.squeeze(data['y'])
  # split the data into training, validation and test sets

  # load pretrained DEC model
  dec = load_snhunters_dec(x_valid, ae_weights, dec_weights, n_clusters, \
    batch_size, lr, momentum)

  cluster_centres = get_cluster_centres(dec)

  query_limit = 128
  for i in range(1,239):
    data = \
      sio.loadmat('../../../../data/snhunters/3pi_20x20_supernova_hunters_batch_%d_signPreserveNorm_detect_misaligned.mat'%(i))
    x = np.nan_to_num(np.reshape(data['X'], (data['X'].shape[0], 400), order='C'))
    y = np.squeeze(data['y'])

    #pca_plot(dec.encoder, x_train, cluster_centres, y=y_train, labels=labels, \
    #         lcolours=lcolours)

    #pca_plot(dec.encoder, x, cluster_centres, y=y, labels=labels, \
    #         lcolours=lcolours)

    y_pred = dec.predict_clusters(x)
  
    cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
      get_cluster_to_label_mapping(y, y_pred, n_classes, n_clusters)

    n_batches = 20
    batch_size = int(len(x)/n_batches)
    print(batch_size)
    for j in range(n_batches):
      x_b = x[j*batch_size:(j+1)*batch_size]
      y_b = y[j*batch_size:(j+1)*batch_size]
      im, cc, ls, cluster_to_label_mapping = \
        get_pairs_auto(dec, x_b, y_b, cluster_centres, cluster_to_label_mapping, \
                       majority_class_fractions, n_clusters)
      
      callbacks = []
  
      model, base_network = train_siamese(dec, cluster_centres, im, cc, ls, \
        epochs=1, split_frac=1.0, callbacks=callbacks)

      y_pred = dec.predict_clusters(x_valid)

      yp = []
      for i in range(len(y_pred)):
        yp.append(cluster_to_label_mapping[y_pred[i]])
      print(f1_score(y_valid, np.array(yp)), percent_fpr(y_valid, np.array(yp), percent))

def volunteer_classification_triplet_selection_test(n):
  np.random.seed(0)
  # constants
  batch_size = 256
  lr         = 0.01
  momentum   = 0.9
  tol        = 0.001
  maxiter         = 2e4
  update_interval = 140

  n_clusters = 10
  n_classes  = 2

  lcolours = ['#D6FF79', '#B0FF92', '#A09BE7', '#5F00BA', '#56CBF9', \
              '#F3C969', '#ED254E', '#CAA8F5', '#D9F0FF', '#46351D']
              
  labels = [str(i) for i in range(n_clusters)]
  
  ae_weights  = '../../../../DEC-keras/results/snh/ae_weights_snh.h5'
  dec_weights = '../../../../DEC-keras/results/snh/%d/DEC_model_final.h5'%n_clusters

  percent = 0.1 # for figure of merit

  # load snhunters data set
  data = sio.loadmat('../../../../data/snhunters/zooniverse_test_set.mat')
  x_valid = np.nan_to_num(np.reshape(data['x'], (data['x'].shape[0], 400), order='F'))
  y_valid  = np.squeeze(data['y'])
  # split the data into training, validation and test sets
  order = np.random.permutation
  x_valid = x_valid[order]
  y_valid = y_valid[order]
  split = int(x_valid.shape[0] / 2.)
  x_test = x_valid[split:]
  y_test = y_valid[split:]
  x_valid = x_valid[:split]
  y_valid = y_valid[:split]

  # load pretrained DEC model
  dec = load_snhunters_dec(x_valid, ae_weights, dec_weights, n_clusters, \
    batch_size, lr, momentum)
  #cluster_centres = get_cluster_centres(dec)
  limit=None
  for i in range(1,n+1):
    data = \
      sio.loadmat('../../../../data/snhunters/3pi_20x20_supernova_hunters_batch_%d_signPreserveNorm_detect_misaligned.mat'%(i))
    x = np.nan_to_num(np.reshape(data['X'], (data['X'].shape[0], 400), order='C'))
    y = np.squeeze(data['y'])

    u, indices = np.unique(x, return_index=True, axis=0)
    x = x[indices][:limit]
    y = y[indices][:limit]
    try:
      X = np.concatenate((X,x))
      Y = np.concatenate((Y,y))
    except UnboundLocalError:
      X = x
      Y = y
  u, indices = np.unique(X, return_index=True, axis=0)
  x = X[indices]
  y = Y[indices]
  del X
  del Y

  #pca_plot(dec.encoder, x, cluster_centres, y=y, \
  #  labels=[str(i) for i in range(n_clusters)], lcolours=[lcolours[0], lcolours[5]])
    
  cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
    get_cluster_to_label_mapping_safe(y, dec.predict_clusters(x), n_classes, n_clusters)
    
  calc_f1_score(y, dec.predict_clusters(x), cluster_to_label_mapping)

  limit = 2048
  metric_log = open('./results/metric_logging.csv', 'w')
  for iter in range(100):
    m = x.shape[0]
    selection = np.random.permutation(m)[:limit]
    pairs, labels = get_pairs_triplet_selection(dec, \
                                                x[selection], \
                                                y[selection],
                                                dec.predict_clusters(x[selection]), \
                                                cluster_to_label_mapping)

    print(pairs.shape)
    print(labels.shape)

    m = pairs.shape[0]
    order = np.random.permutation(m)
    pairs = pairs[order]
    labels = labels[order]
    
    callbacks = [CSVLogger('./results/training.log')]
    _, val_split = train_siamese_triplet_selection(dec, \
                                                   pairs, \
                                                   labels, \
                                                   'sgd', \
                                                   epochs=10, \
                                                   callbacks=callbacks, \
                                                   split_frac=.75)

    dec.clustering(x[selection], save_dir='./results/%d/'%(iter))
    
    #update the cluster to label mapping
    cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
      get_cluster_to_label_mapping_safe(y, dec.predict_clusters(x), n_classes, n_clusters)
    
    val_f1 = calc_f1_score(y[selection][val_split:], \
                           dec.predict_clusters(x[selection][val_split[:]]), \
                           cluster_to_label_mapping))
                           
    train_f1 = calc_f1_score(y[selection][:val_split], \
                             dec.predict_clusters(x[selection][:val_split]), \
                             cluster_to_label_mapping))
    
    unlabelled_f1 = calc_f1_score(y_valid, \
                             dec.predict_clusters(x_valid), \
                             cluster_to_label_mapping))
                             
    c_purities = np.mean(np.array(np.nan_to_num(majority_class_fractions)) \
               * (np.array(n_assigned_list) \
               / (float(limit))))

    metrics = [val_f1,train_f1,unlabelled_f1,c_purities]
    #cluster_centres = get_cluster_centres(dec)
    #pca_plot(dec.encoder, x, cluster_centres, y=y, \
    #  labels=[str(i) for i in range(n_clusters)], lcolours=[lcolours[0], lcolours[5]])
    metric_log.write((',').join(metrics))
  metric_log.close()
  cluster_to_label_mapping_test, n_assigned_list, majority_class_fractions = \
    get_cluster_to_label_mapping_safe(y_test, dec.predict_clusters(x_test), n_classes, n_clusters)

  unlabelled_test_f1 = calc_f1_score(y_test, \
                                     dec.predict_clusters(x_test), \
                                     cluster_to_label_mapping))
  print(unlabelled_test_f1)
  unlabelled_test_f1 = calc_f1_score(y_test, \
                                     dec.predict_clusters(x_test), \
                                     cluster_to_label_mapping_test))
  print(unlabelled_test_f1)                                    
  #cluster_centres = get_cluster_centres(dec)
  #pca_plot(dec.encoder, x_test, cluster_centres, y=y_test, \
  #  labels=[str(i) for i in range(n_clusters)], lcolours=[lcolours[0], lcolours[5]])

def main():
  #volunteer_classification_test()
  volunteer_classification_triplet_selection_test(10)

if __name__ == '__main__':
  main()
