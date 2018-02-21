import sys
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

sys.path.insert(0,'../../../../DEC-keras')
from DEC import DEC
from datasets import load_mnist

sys.path.insert(0,'../../')
from dissolving_utils import get_cluster_centres, get_cluster_to_label_mapping
from dissolving_utils import pca_plot, FrameDumpCallback

sys.path.insert(0,'../')
from siamese_utils import get_pairs_auto, train_siamese

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
  sample_frac = 0.2
  split = int(sample_frac*x.shape[0])
  x_train = x[:split]
  y_train = y[:split]
  x_test  = x[split:]
  y_test  = y[split:]

  dec = load_mnist_dec(x, ae_weights, dec_weights, n_clusters, \
    batch_size, lr, momentum)
    
  y_pred = dec.predict_clusters(x_train)

  cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
    get_cluster_to_label_mapping(y_train, y_pred, n_classes, n_clusters)

  cluster_centres = get_cluster_centres(dec)

  y_plot = np.array(y[:],dtype='int')
  y_plot[split:] = -1

  pca_plot(dec.encoder, x, cluster_centres, y=y_plot, labels=labels, \
             lcolours=lcolours)

  im, cc, ls, cluster_to_label_mapping = \
    get_pairs_auto(dec, x_train, y_train, cluster_centres, \
      cluster_to_label_mapping, majority_class_fractions, n_clusters)

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

  model, base_network = train_siamese(dec, x, cluster_centres, im, cc, ls, \
    epochs=100, split_frac=0.75, callbacks=callbacks)

  y_pred =dec.predict_clusters(x_train)
  
  cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
    get_cluster_to_label_mapping(y_train, y_pred, n_classes, n_clusters)

  y_pred =dec.predict_clusters(x_test)
  
  cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
    get_cluster_to_label_mapping(y_test, y_pred, n_classes, n_clusters)

  pca_plot(dec.encoder, x, cluster_centres, y=y_plot, labels=labels, \
             lcolours=lcolours)

if __name__ == '__main__':
  main()
