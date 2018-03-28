import sys
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

sys.path.insert(0,'../../../../DEC-keras')
from DEC import DEC, cluster_acc
from datasets import load_mnist

sys.path.insert(0,'../../')
from dissolving_utils import get_cluster_centres, get_cluster_to_label_mapping
from dissolving_utils import pca_plot, FrameDumpCallback

sys.path.insert(0,'../')
from siamese_utils import get_pairs_auto, get_pairs_auto_with_noise, train_siamese, train_siamese_online

def load_mnist_dec(x, ae_weights, dec_weights, n_clusters, batch_size, lr, \
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
  n_classes  = 10

  lcolours = ['#D6FF79', '#B0FF92', '#A09BE7', '#5F00BA', '#56CBF9', \
              '#F3C969', '#ED254E', '#CAA8F5', '#D9F0FF', '#46351D']
  labels = [str(i) for i in range(n_clusters)]
  
  ae_weights  = '../../../../DEC-keras/results/mnist/ae_weights.h5'
  dec_weights = '../../../../DEC-keras/results/mnist/%d/DEC_model_final.h5'%n_clusters
  
  # load mnist data set
  x, y = load_mnist()
  # split the data into training, validation and test sets
  m = x.shape[0]
  m = m - 20000
  sample_frac = 0.01
  split = int(sample_frac*m)
  print(split)
  x_train = x[:split]
  y_train = y[:split]
  x_valid = x[50000:60000]
  y_valid = y[50000:60000]
  x_test  = x[60000:]
  y_test  = y[60000:]

  # load pretrained DEC model
  dec = load_mnist_dec(x, ae_weights, dec_weights, n_clusters, \
    batch_size, lr, momentum)

  # predict training set cluster assignments
  y_pred = dec.predict_clusters(x_train)

  # inspect the clustering and simulate volunteer labelling of random sample (the training set)
  cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
    get_cluster_to_label_mapping(y_train, y_pred, n_classes, n_clusters)
  print(cluster_acc(y_train, y_pred))
  y_valid_pred = dec.predict_clusters(x_valid)
  print(cluster_acc(y_valid, y_valid_pred))
  
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
  print(cluster_acc(y_valid, y_pred))
  #pca_plot(dec.encoder, x_valid, cluster_centres, y=y_valid, labels=labels, \
  #           lcolours=lcolours)

  y_pred = dec.predict_clusters(x[:m])
  print(np.argmin(majority_class_fractions))

  for j in range(1,6):
    selection = np.where(y_pred[j*split:(j+1)*split] == np.argmin(majority_class_fractions))
    x_train = np.concatenate((x_train, x[:m][j*split:(j+1)*split][selection]))
    y_train = np.concatenate((y_train, y[:m][j*split:(j+1)*split][selection]))
  
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
    print(cluster_acc(y_valid, y_pred))
    #pca_plot(dec.encoder, x_valid, cluster_centres, y=y_valid, labels=labels, \
    #          lcolours=lcolours)

if __name__ == '__main__':
  main()
