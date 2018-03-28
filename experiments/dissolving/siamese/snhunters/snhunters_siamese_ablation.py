import sys
import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.initializers import Initializer
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.engine.topology import Layer
from keras import backend as K
from keras.models import load_model

sys.path.insert(0,'../../../../DEC-keras')
from DEC import DEC, ClusteringLayer, cluster_acc

sys.path.insert(0,'../../')
from dissolving_utils import get_cluster_centres, get_cluster_to_label_mapping
from dissolving_utils import pca_plot, FrameDumpCallback, percent_fpr
from dissolving_utils import MappingLayer, MapInitializer
from dissolving_utils import load_dec

sys.path.insert(0,'../')
from siamese_utils import get_pairs_auto, get_pairs_auto_with_noise, train_siamese, train_siamese_online

lcolours = ['#D6FF79', '#B0FF92', '#A09BE7', '#5F00BA', '#56CBF9', \
            '#F3C969', '#ED254E', '#CAA8F5', '#D9F0FF', '#46351D']

# DEC constants from DEC paper
batch_size = 256
lr         = 0.01
momentum   = 0.9
tol        = 0.001
maxiter    = 2e4
update_interval = 140
n_clusters = 10 # number of clusters to use
n_classes  = 2  # number of classes

def calc_f1_score(y_true, predicted_clusters, cluster_to_label_mapping):
  y_pred = []
  for i in range(len(y_true)):
    y_pred.append(cluster_to_label_mapping[predicted_clusters[i]])
  return f1_score(y_true, np.array(y_pred))

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

def supervised_batch_size_test():
  """
    This ablation test removes the siamese network and attempts to update the
    network using supervised learning.  The experiment is run for multiple sample
    fractions.
  """

  trials = {}
  n_trials = 5
  sample_fracs = [0.01, 0.05, 0.1, 0.25, 0.5]
  #sample_fracs = [0.5]
  baselines = []
  for sample_frac in sample_fracs:
    x, y, m, split, x_train, y_train, x_valid, y_valid, x_test, y_test = \
        load_data(sample_frac)
  
    # load the pretrained DEC model for Supernova Hunters
    ae_weights  = '../../../../DEC-keras/results/snh/ae_weights_snh.h5'
    dec_weights = '../../../../DEC-keras/results/snh/%d/DEC_model_final.h5'%n_clusters
    
    dec = load_dec(x, ae_weights, dec_weights, n_clusters, batch_size, lr, momentum)
    x_train = x[:split]
    y_train = y[:split]
    y_pred = dec.predict_clusters(x_train)
    # inspect the clustering and simulate volunteer labelling of random sample (the training set)
    cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
      get_cluster_to_label_mapping_safe(y_train, y_pred, n_classes, n_clusters)
    baselines.append(calc_f1_score(y_valid, dec.predict_clusters(x_valid), cluster_to_label_mapping))

  for sample_frac in sample_fracs:
    FoMs = []
    Errs = []
    n_labelled = []
    for k in range(n_trials):
      x, y, m, split, x_train, y_train, x_valid, y_valid, x_test, y_test = \
        load_data(sample_frac)
  
      # load the pretrained DEC model for Supernova Hunters
      ae_weights  = '../../../../DEC-keras/results/snh/ae_weights_snh.h5'
      dec_weights = '../../../../DEC-keras/results/snh/%d/DEC_model_final.h5'%n_clusters
    
      dec = load_dec(x, ae_weights, dec_weights, n_clusters, batch_size, lr, momentum)
      foms = []
      for i in range(int(1/sample_frac)):
        x_train = x[i*split:(i+1)*split]
        y_train = y[i*split:(i+1)*split]
        #x_train = x[:(i+1)*split]
        #y_train = y[:(i+1)*split]
        # predict training set cluster assignments
        y_pred = dec.predict_clusters(x[:(i+1)*split])
        # inspect the clustering and simulate volunteer labelling of random sample (the training set)
        cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
          get_cluster_to_label_mapping_safe(y[:(i+1)*split], y_pred, n_classes, n_clusters)

        a = Input(shape=(400,)) # input layer
        q = dec.model(a)
        pred = MappingLayer(cluster_to_label_mapping, output_dim=n_classes, kernel_initializer=MapInitializer(cluster_to_label_mapping, n_classes))(q)
        model = Model(inputs=a, outputs=pred)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        model.fit(x_train, np_utils.to_categorical(y_train, 2), epochs=1, batch_size=256)

        y_pred = dec.predict_clusters(x[:(i+1)*split])
        cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
          get_cluster_to_label_mapping_safe(y[:(i+1)*split], y_pred, n_classes, n_clusters)
      
        foms.append(calc_f1_score(y_valid, dec.predict_clusters(x_valid), cluster_to_label_mapping))
        if k == 0:
          n_labelled.append((i+1)*split)
      FoMs.append(foms)
      #Errs.append(np.std(foms))
      trials[sample_frac] = (n_labelled, np.array(FoMs))

  #print(calc_f1_score(y_valid, dec.predict_clusters(x_valid), cluster_to_label_mapping))
  #print(calc_f1_score(y_test, dec.predict_clusters(x_test), cluster_to_label_mapping))
  #print(trials[sample_frac])
  #print(trials[sample_frac][1].shape)

  pickle.dump(trials, open('supervised_batch_size_test_epochs1_trials5_safe.pkl','wb'))
  
  plt.plot([0,6000],[0.924951892239, 0.924951892239], 'k--')
  for i, key in enumerate(trials.keys()):
    plt.plot([0,6000],[baselines[i], baselines[i]], '--', color=lcolours[i+2])
    plt.plot(trials[key][0], np.mean(trials[key][1], axis=0), '-', color=lcolours[i+2])
    plt.errorbar(trials[key][0], np.mean(trials[key][1], axis=0), yerr=np.std(trials[key][1], axis=0), fmt='o', color=lcolours[i+2], label=str(key))
  plt.ylabel('validation set F1-score')
  plt.xlabel('number of labelled examples')
  plt.ylim(0,1)
  plt.xlim(0,6000)
  #
  plt.legend()
  #plt.savefig('siamese_ablation_test_concat_epochs1_trials5.pdf')
  plt.show()

def supervised_volunteer_classifications_test(selection_func, sample_frac):
  """
    This ablation test removes the siamese network and attempts to update the
    network using supervised learning.  The experiment is run for multiple sample
    fractions.
  """
  
  X, Y, _, _, x_train, y_train, x_valid, y_valid, x_test, y_test = \
    load_data(sample_frac)

  # load the pretrained DEC model for Supernova Hunters
  ae_weights  = '../../../../DEC-keras/results/snh/ae_weights_snh.h5'
  dec_weights = '../../../../DEC-keras/results/snh/%d/DEC_model_final.h5'%n_clusters

  dec = load_dec(X, ae_weights, dec_weights, n_clusters, batch_size, lr, momentum)

  m = 0
  foms = []
  ms = []
  for i in range(0,100):
    data = sio.loadmat('../../../../data/3pi_20x20_supernova_hunters_batch_%d_signPreserveNorm.mat'%(i+1))
    try:
      x = np.concatenate((x, np.nan_to_num(np.reshape(data['X'], (data['X'].shape[0], 400), order='C'))))
      y = np.concatenate((y, np.squeeze(data['y'])))
    except UnboundLocalError:
      x = np.nan_to_num(np.reshape(data['X'], (data['X'].shape[0], 400), order='C'))
      y = np.squeeze(data['y'])
      batch = x.shape[0]
    split = int(sample_frac*batch)
    for j in range(0,int(batch/split)):
      #x_train = x[:i*batch+(j+1)*split]
      #y_train = y[:i*batch+(j+1)*split]
      x_train, y_train = selection_func(x, y, i, j, batch, split)
      
      m = x_train.shape[0]
      # predict training set cluster assignments
      ##y_pred = dec.predict_clusters(X)
      y_pred = dec.predict_clusters(x_train)
      # inspect the clustering and simulate volunteer labelling of random sample (the training set)
      ##cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
      ##  get_cluster_to_label_mapping_safe(Y, y_pred, n_classes, n_clusters)
      cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
        get_cluster_to_label_mapping_safe(y_train, y_pred, n_classes, n_clusters)
      a = Input(shape=(400,)) # input layer
      q = dec.model(a)
      pred = MappingLayer(cluster_to_label_mapping, output_dim=n_classes, kernel_initializer=MapInitializer(cluster_to_label_mapping, n_classes))(q)
      model = Model(inputs=a, outputs=pred)
      model.compile(loss='categorical_crossentropy', optimizer='adam')

      model.fit(x_train, np_utils.to_categorical(y_train, 2), epochs=1, batch_size=256)

      y_pred = dec.predict_clusters(X)
      cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
        get_cluster_to_label_mapping_safe(Y, y_pred, n_classes, n_clusters)
      
      foms.append(calc_f1_score(y_test, dec.predict_clusters(x_test), cluster_to_label_mapping))
      ms.append(m)
      print(foms)
      print(ms)
  print(foms)
  print(ms)

  plt.plot(ms,foms,'o')
  plt.plot(ms,foms,'-')
  plt.ylabel('test set F1-score')
  plt.xlabel('number of labelled examples')
  plt.ylim(0,1)
  #plt.xlim(0,6000)
  #
  plt.legend()
  #plt.savefig('siamese_ablation_test_concat_epochs1_trials5.pdf')
  plt.show()

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

def test_get_cluster_to_label_mapping_safe():
  sample_fracs = [0.01, 0.05, 0.1, 0.25, 0.5]
  for sample_frac in sample_fracs:
    x, y, m, split, x_train, y_train, x_valid, y_valid, x_test, y_test = \
      load_data(sample_frac)
    # load the pretrained DEC model for Supernova Hunters
    ae_weights  = '../../../../DEC-keras/results/snh/ae_weights_snh.h5'
    dec_weights = '../../../../DEC-keras/results/snh/%d/DEC_model_final.h5'%n_clusters
    
    dec = load_dec(x, ae_weights, dec_weights, n_clusters, batch_size, lr, momentum)
    x_train = x[:split]
    y_train = y[:split]
    y_pred = dec.predict_clusters(x_train)
    # inspect the clustering and simulate volunteer labelling of random sample (the training set)
    cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
      get_cluster_to_label_mapping_safe(y_train, y_pred, n_classes, n_clusters)
  
def plot_supervised_batch_size_test_concat():

  trials = pickle.load(open('supervised_batch_size_test_concat_epochs1_trials5.pkl','rb'))
  plt.plot([0,6000],[0.924951892239, 0.924951892239], 'k--')
  for i, key in enumerate(trials.keys()):
    #plt.plot([0,6000],[baselines[i], baselines[i]], '--', color=lcolours[i+2])
    plt.plot(trials[key][0], np.mean(trials[key][1], axis=0), '-', color=lcolours[i+2])
    plt.errorbar(trials[key][0], np.mean(trials[key][1], axis=0), yerr=np.std(trials[key][1], axis=0), fmt='o', color=lcolours[i+2], label=str(key))
  plt.ylabel('validation set F1-score')
  plt.xlabel('number of labelled examples')
  plt.ylim(-0.05,1.05)
  plt.xlim(0,5500)
  #plt.show()
  plt.legend()
  #plt.savefig('siamese_ablation_test_concat_epochs1_trials5.pdf')
  plt.show()

def no_selection(x, y, *args):
  # i, j, batch, split_frac
  return x[:args[0]*args[2]+(args[1]+1)*args[3]], y[:args[0]*args[2]+(args[1]+1)*args[3]]

def select_misses_only():

def main():
  #supervised_batch_size_test()
  #plot_supervised_batch_size_test_concat()
  #test_get_cluster_to_label_mapping_safe()
  supervised_volunteer_classifications_test(no_selection, sample_frac=0.5)

if __name__ == '__main__':
  main()
