import numpy as np

np.random.seed(1337)

import matplotlib.pyplot as plt

from utils import loadMNIST, clusterData, calculateAccuracy
from utils import buildModel, runTrials, getMACResultsDB
from utils import calculateClassDistribution, calculateLabellingAccuracy

from keras.utils import np_utils

def experiment2aNoiseLabelling(X, y, clustering, n_classes, \
  noise=None, intelligent_noise=None):

  assert noise == None or intelligent_noise == None
  
  n_clusters = clustering.n_clusters
  
  for cluster in range(n_clusters):
    cluster_indices = np.where(clustering.labels_ == cluster)[0]
    n_assigned_examples = cluster_indices.shape[0]
    cluster_labels = y[cluster_indices]
    cluster_label_fractions = np.mean(cluster_labels, axis=0)
    dominant_cluster_class = np.argmax(cluster_label_fractions)
    classes = list(range(n_classes))
    if noise and np.random.rand() < noise:
      classes.remove(dominant_cluster_class)
      dominant_cluster_class = np.random.choice(classes)
    if intelligent_noise and np.random.rand() < intelligent_noise:
      ordered = cluster_label_fractions.argsort()
      classes = np.array(classes)[ordered][1:]
      for c in classes:
        if np.random.rand() < cluster_label_fractions[c]:
          dominant_cluster_class = c
          break
      if dominant_cluster_class == np.argmax(cluster_label_fractions):
        dominant_cluster_class = classes[0]
    print(cluster, n_assigned_examples, dominant_cluster_class, \
          cluster_label_fractions[dominant_cluster_class])
    # assign labels based on dominant class
    x = X[cluster_indices]
    l = np.zeros((x.shape[0], 10))
    l[:,dominant_cluster_class] += 1
    try:
      x_labelled = np.concatenate((x_labelled, x))
      labels = np.concatenate((labels, l))
      labelled_indices = np.concatenate((labelled_indices, cluster_indices))
    except NameError:
      x_labelled = x
      labels = l
      labelled_indices = cluster_indices
        
  print(x_labelled.shape)
  print(labels.shape)

  m = x_labelled.shape[0]
  order = np.random.permutation(m)
  x_labelled = x_labelled[order]
  x_labelled = x_labelled
  labels = labels[order]
  labelled_indices = labelled_indices[order]

  unlabelled_indices = np.array([x for x in range(X.shape[0]) \
                                 if x not in labelled_indices])

  return x_labelled, labels, labelled_indices, unlabelled_indices

def experiment2bNoiseLabelling(X, y, clustering, n_classes, fraction, \
  noise=None, intelligent_noise=None):

  assert noise == None or intelligent_noise == None
  
  n_clusters = clustering.n_clusters
  
  for cluster in range(n_clusters):
    cluster_indices = np.where(clustering.labels_ == cluster)[0]
    n_assigned_examples = cluster_indices.shape[0]
    cluster_labels = y[cluster_indices]
    cluster_label_fractions = np.mean(cluster_labels, axis=0)
    dominant_cluster_class = np.argmax(cluster_label_fractions)
    if noise and np.random.rand() < noise:
      classes.remove(dominant_cluster_class)
      dominant_cluster_class = np.random.choice(classes)
    if intelligent_noise and np.random.rand() < intelligent_noise:
      ordered = cluster_label_fractions.argsort()
      classes = np.array(classes)[ordered][1:]
      for c in classes:
        if np.random.rand() < cluster_label_fractions[c]:
          dominant_cluster_class = c
          break
      if dominant_cluster_class == np.argmax(cluster_label_fractions):
        dominant_cluster_class = classes[0]
    print(cluster, n_assigned_examples, dominant_cluster_class, \
          cluster_label_fractions[dominant_cluster_class])
    if cluster_label_fractions[dominant_cluster_class] >= fraction:
      x = X[cluster_indices]
      l = np.zeros((x.shape[0], 10))
      l[:,dominant_cluster_class] += 1
      try:
        x_labelled = np.concatenate((x_labelled, x))
        labels = np.concatenate((labels, l))
        labelled_indices = np.concatenate((labelled_indices, cluster_indices))
      except NameError:
        x_labelled = x
        labels = l
        labelled_indices = cluster_indices
        
  print(x_labelled.shape)
  print(labels.shape)

  m = x_labelled.shape[0]
  order = np.random.permutation(m)
  x_labelled = x_labelled[order]
  x_labelled = x_labelled
  labels = labels[order]
  labelled_indices = labelled_indices[order]

  unlabelled_indices = np.array([x for x in range(X.shape[0]) \
                                 if x not in labelled_indices])

  return x_labelled, labels, labelled_indices, unlabelled_indices

def experiment2aNoise(filename):
  image_dim = 28
  n_classes = 10
  n_trials = 5
  epochs = 20
  batch_size = 500
  
  db = getMACResultsDB()
  
  x_train, y_train, x_train_flattened, x_test, y_test = loadMNIST()

  # calculate performance on gold labels
  y_train = np_utils.to_categorical(y_train, n_classes)
  y_test = np_utils.to_categorical(y_test, n_classes)
  
  clustering = clusterData(x_train_flattened)

  # experiment 2a labelling accuracy
  labelling_accuracies_2a = []
  labelling_accuracies_2a_errors = []
  labelling_accuracies_2a_2 = []
  labelling_accuracies_2a_2_errors = []
  # experiment 2a machine accuracy
  machine_accuracies_2a = []
  machine_accuracies_2a_errors = []
  machine_accuracies_2a_2 = []
  machine_accuracies_2a_2_errors = []
  
  noise_levels = np.arange(0,1.1,0.1)

  collection_name = 'experiment2d'
  collection = db[collection_name]

  for noise in noise_levels:
    l_results = []
    l_results_2 = []
    m_results = []
    m_results_2 = []
    for i in range(n_trials):
      # experiment 2a with random noise
      experiment_name = \
        'experiment 2d - experiment2a random noise %.2lf trial %d' % (noise, i)
      try:
        assert collection_name in db.collection_names()
        doc = db[collection_name].find({'name':experiment_name})[0]
        l_results.append(doc['labelling accuracy'])
        m_results.append(doc['machine accuracy'])
      except (AssertionError, IndexError):
        x_labelled, labels, labelled_indices, unlabelled_indices = \
          experiment2aNoiseLabelling(x_train, y_train, clustering, \
          n_classes, noise=noise)
        l_results.append(calculateLabellingAccuracy(y_train[labelled_indices], labels))
        r = runTrials(x_labelled, labels, x_test, y_test, 1, n_classes, \
          data='mnist_experiment2d_noise%.2lf_trial%d'%(noise, i))
        m_results.append(r[0])
        doc = {
                'name':experiment_name,
                'm': labels.shape[0],
                'noise': noise,
                'trial': i,
                'labelling accuracy': \
                  calculateLabellingAccuracy(y_train[labelled_indices], labels),
                'machine accuracy': r[0],
                'training set class distribution': \
                  calculateClassDistribution(labels).tolist()
              }
        collection.insert_one(doc)

      # experiment 2a with weighted noise
      experiment_name = \
        'experiment 2d - experiment2a class weighted noise %.2lf trial %d' \
          % (noise, i)
      try:
        assert collection_name in db.collection_names()
        doc = db[collection_name].find({'name':experiment_name})[0]
        l_results_2.append(doc['labelling accuracy'])
        m_results_2.append(doc['machine accuracy'])
      except (AssertionError, IndexError):
        x_labelled, labels, labelled_indices, unlabelled_indices = \
          experiment2aNoiseLabelling(x_train, y_train, clustering, \
          n_classes, intelligent_noise=noise)
        l_results_2.append(calculateLabellingAccuracy(y_train[labelled_indices], labels))
        r = runTrials(x_labelled, labels, x_test, y_test, 1, n_classes, \
          data='mnist_experiment2d_class_weighted_noise%.2lf_trial%d'%(noise, i))
        m_results_2.append(r[0])
        doc = {
                'name':experiment_name,
                'm': labels.shape[0],
                'noise': noise,
                'trial': i,
                'labelling accuracy': \
                  calculateLabellingAccuracy(y_train[labelled_indices], labels),
                'machine accuracy': r[0],
                'training set class distribution': \
                  calculateClassDistribution(labels).tolist()
              }
        collection.insert_one(doc)
        
    labelling_accuracies_2a.append(np.mean(l_results))
    labelling_accuracies_2a_errors.append(np.std(l_results))
    labelling_accuracies_2a_2.append(np.mean(l_results_2))
    labelling_accuracies_2a_2_errors.append(np.std(l_results_2))

    machine_accuracies_2a.append(np.mean(m_results))
    machine_accuracies_2a_errors.append(np.std(m_results))
    machine_accuracies_2a_2.append(np.mean(m_results_2))
    machine_accuracies_2a_2_errors.append(np.std(m_results_2))

  cursor = db['experiment2a'].find({'name':'majority class label assignment'})
  for doc in cursor:
    majority_class_benchmark = doc['labelling accuracy']
  
  fig, ax = plt.subplots()
  ax.plot(np.arange(-0.02,1.03,0.01), \
    np.ones((105))*majority_class_benchmark,'k--')

  ax.errorbar(noise_levels, labelling_accuracies_2a, \
      yerr=labelling_accuracies_2a_errors, fmt='o', mfc='None', \
      color='#B8336A', label='majority class - random noise')

  ax.errorbar(noise_levels, labelling_accuracies_2a_2, \
      yerr=labelling_accuracies_2a_2_errors, fmt='o', mfc='None', \
      color='#726DA8', label='majority class - class weighted noise', zorder=100)

  ax.set_xlabel('labelling noise')
  ax.set_ylabel('labelling accuracy')
  ax.set_ylim(-2,100)
  ax.set_xlim(-0.02,1.03)
  plt.legend(loc='lower left')
  #plt.show()
  plt.savefig(filename+'labelling_noise.pdf')
  plt.savefig(filename+'labelling_noise.png')

  cursor = db['experiment2a'].find({'name' : 'gold benchmark'})
  for doc in cursor:
    gold_benchmark = doc['accuracy']
    gold_benchmark_error = doc['error']

  cursor = db['experiment2a'].find({'name' : 'majority class label assignment'})
  for doc in cursor:
    majority_class_benchmark = doc['accuracy']
    majority_class_error = doc['error']
  
  fig, ax = plt.subplots()

  ax.plot(np.arange(-0.02,1.03,0.01), np.ones((105,))*gold_benchmark, \
          color='#726DA8', label='gold benchmark')
  ax.axhspan(gold_benchmark-gold_benchmark_error, \
             gold_benchmark+gold_benchmark_error, \
             facecolor='#726DA8', alpha=0.5)

  ax.plot(np.arange(-0.02,1.03,0.01), np.ones((105,))*majority_class_benchmark, \
          color='#A0D2DB', label='majority class benchmark')
  ax.axhspan(majority_class_benchmark-majority_class_error, \
             majority_class_benchmark+majority_class_error, \
             facecolor='#A0D2DB', alpha=0.5)

  ax.errorbar(noise_levels, machine_accuracies_2a, \
      yerr=machine_accuracies_2a_errors, fmt='o', mfc='None', \
      color='#B8336A', label='majority class - random noise')

  ax.errorbar(noise_levels, machine_accuracies_2a_2, \
      yerr=machine_accuracies_2a_2_errors, fmt='o', mfc='None', \
      color='#726DA8', label='majority class - class weighted noise', zorder=100)

  ax.set_xlabel('labelling noise')
  ax.set_ylabel('machine accuracy')
  ax.set_ylim(-2,100)
  ax.set_xlim(-0.02,1.03)
  plt.legend(loc='lower left')
  #plt.show()
  plt.savefig(filename+'_machine_accuracy.pdf')
  plt.savefig(filename+'_machine_accuracy.png')

def experiment2d():
  
  db = getMACResultsDB()
  collection = db['experiment2d']

  experiment2aNoise('experiment2d_2a_noise')

  """
  # experiment 2b
  fig, ax = plt.subplots()
  fractions = [0.11, 0.2, 0.25, 0.3, 0.4, 0.5, \
               0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]

  for noise in noise_levels:
    for fraction in fractions:
      # experiment 2b with random noise
      x_labelled, labels, labelled_indices, unlabelled_indices = \
        experiment2bNoise(x_train, y_train, clustering, fraction, \
        n_classes, noise=noise)

      result = calculateLabellingAccuracy(y_train[labelled_indices], labels)
      ax.plot(noise_levels, result, 'o', color='#726DA8')
  plt.show()
  """
def main():
  experiment2d()

if __name__ == '__main__':
  main()
