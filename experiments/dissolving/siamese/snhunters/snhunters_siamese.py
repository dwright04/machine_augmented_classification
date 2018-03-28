import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

sys.path.insert(0,'../../../../DEC-keras')
from DEC import DEC

sys.path.insert(0,'../../')
from dissolving_utils import get_cluster_centres, get_cluster_to_label_mapping
from dissolving_utils import pca_plot, FrameDumpCallback, percent_fpr

sys.path.insert(0,'../')
from siamese_utils import get_pairs_auto, get_pairs_auto_with_noise, train_siamese, train_siamese_online

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

def main():

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

if __name__ == '__main__':
  main()
