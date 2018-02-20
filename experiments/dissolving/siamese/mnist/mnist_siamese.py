import sys

from keras.optimizers import SGD

sys.path.insert(0,'../../../../DEC-keras')
from DEC import DEC
from datasets import load_mnist

sys.path.insert(0,'../../')
from dissolving_utils import get_cluster_centres, get_cluster_to_label_mapping

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
  
  ae_weights  = '../../../../DEC-keras/results/mnist/ae_weights.h5'
  dec_weights = '../../../../DEC-keras/results/mnist/%d/DEC_model_final.h5'%n_clusters
  
  # load mnist data set
  x, y = load_mnist()
  sample_frac = 0.05
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




if __name__ == '__main__':
  main()
