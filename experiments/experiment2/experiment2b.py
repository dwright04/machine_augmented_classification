import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)

from utils import loadMNIST, clusterData, calculateAccuracy
from utils import buildModel, runTrials, getMACResultsDB
from utils import calculateClassDistribution, calculateLabellingAccuracy

from keras.utils import np_utils

def experiment2bLabelleingMethod(X, y, clustering, n_classes, fraction):

  n_clusters = clustering.n_clusters
  
  for cluster in range(n_clusters):
    cluster_indices = np.where(clustering.labels_ == cluster)[0]
    n_assigned_examples = cluster_indices.shape[0]
    cluster_labels = y[cluster_indices]
    cluster_label_fractions = np.mean(cluster_labels, axis=0)
    dominant_cluster_class = np.argmax(cluster_label_fractions)
    print(cluster, n_assigned_examples, dominant_cluster_class, \
          cluster_label_fractions[dominant_cluster_class])
    if cluster_label_fractions[dominant_cluster_class] >= fraction:
      x = X[cluster_indices]
      l = np.zeros((x.shape[0], n_classes))
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

def experiment2b():

  image_dim = 28
  n_classes = 10
  n_trials = 5
  epochs = 20
  batch_size = 500

  db = getMACResultsDB()
  collection = db['experiment2b']
  
  x_train, y_train, x_train_flattened, x_test, y_test = loadMNIST()

  y_train = np_utils.to_categorical(y_train, n_classes)
  y_test = np_utils.to_categorical(y_test, n_classes)

  clustering = clusterData(x_train_flattened)
  
  fractions = [0.11, 0.2, 0.25, 0.3, 0.4, 0.5, \
               0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
  
  results = []
  for fraction in fractions:
    x_labelled, labels, labelled_indices, unlabelled_indices = \
     experiment2bLabelleingMethod(x_train, y_train, clustering,
                                  n_classes, fraction)

    print(labels.shape[0])

    result = \
      runTrials(x_labelled, labels, x_test, y_test, n_trials, n_classes, \
                data='mnist_experiment2b_fraction%.2lf'%(fraction), \
                epochs=epochs)

    results.append(result)
    
    doc = {
          'name': 'majority class label assignment by fraction',
          'm': labels.shape[0],
          'fraction': fraction,
          'accuracy': result[0],
          'error': result[1],
          'trials accuracy': result[2],
          'labelling accuracy': \
            calculateLabellingAccuracy(y_train[labelled_indices], labels),
          'training set class distribution': \
            calculateClassDistribution(labels).tolist()
        }
        
    collection.insert_one(doc)
  
  return fractions, results

def plotMachineAccuracy(accuracies, errors, fractions, filename, \
  experiment='experiment2a'):

  db = getMACResultsDB()
  cursor = db[experiment].find({'name' : 'gold benchmark'})
  for doc in cursor:
    gold_benchmark = doc['accuracy']
    gold_benchmark_error = doc['error']

  cursor = db[experiment].find({'name' : 'majority class label assignment'})
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

  for i, fraction in enumerate(fractions):
    ax.errorbar(fraction, accuracies[i], yerr=errors[i], \
                 fmt='o', color='#B8336A')
  
  ax.set_ylim(0,100)
  ax.set_xlim(-0.02,1.02)
  ax.set_xlabel('majority class cluster proportion')
  ax.set_ylabel('machine test set accuracy')
  plt.legend(loc='lower right')
  #plt.show()
  plt.savefig(filename+'.pdf')
  plt.savefig(filename+'.png')

def plotLabellingAccuracy(accuracies, fractions, filename, \
  experiment='experiment2a'):

  db = getMACResultsDB()
  cursor = db[experiment].find({'name' : 'gold benchmark'})
  for doc in cursor:
    gold_benchmark = doc['labelling accuracy']

  cursor = db[experiment].find({'name' : 'majority class label assignment'})
  for doc in cursor:
    majority_class_benchmark = doc['labelling accuracy']
  
  fig, ax = plt.subplots()

  ax.plot(np.arange(-0.02,1.03,0.01), np.ones((105,))*gold_benchmark, \
          color='#726DA8', label='gold benchmark')

  ax.plot(np.arange(-0.02,1.03,0.01), np.ones((105,))*majority_class_benchmark, \
          color='#A0D2DB', label='majority class benchmark')

  for i, fraction in enumerate(fractions):
    ax.plot(fraction, accuracies[i], \
            'o', color='#B8336A')
  
  ax.set_ylim(0,102)
  ax.set_xlim(-0.02,1.02)
  ax.set_xlabel('majority class cluster proportion')
  ax.set_ylabel('labelling accuracy')
  plt.legend(loc='lower right')
  #plt.show()
  plt.savefig(filename+'.pdf')
  plt.savefig(filename+'.png')

def plotNumberTrainingExamples(number_training_examples, fractions, filename, \
  experiment='experiment2a'):

  db = getMACResultsDB()
  cursor = db[experiment].find({'name' : 'gold benchmark'})
  for doc in cursor:
    gold_benchmark = doc['m']
  
  fig, ax = plt.subplots()

  ax.plot(np.arange(-0.02,1.03,0.01), np.ones((105,))*gold_benchmark, \
          color='#726DA8', label='total training examples')

  for i, fraction in enumerate(fractions):
    ax.plot(fraction, number_training_examples[i], \
            'o', color='#B8336A')
  
  ax.set_ylim(0,10200)
  ax.set_xlim(-0.02,1.02)
  ax.set_xlabel('majority class cluster proportion')
  ax.set_ylabel('number of training examples')
  plt.legend(loc='lower left')
  #plt.show()
  plt.savefig(filename+'.pdf')
  plt.savefig(filename+'.png')

def main():
  db = getMACResultsDB()
  experiment_name = 'experiment2b'
  accuracies = []
  labelling_accuracies = []
  number_training_examples = []
  errors = []
  try:
    assert experiment_name in db.collection_names()
    cursor = db[experiment_name].find()
    fractions = []
    for doc in cursor:
      print(doc)
      accuracies.append(doc['accuracy'])
      labelling_accuracies.append(doc['labelling accuracy'])
      errors.append(doc['error'])
      fractions.append(doc['fraction'])
      number_training_examples.append(doc['m'])
  except AssertionError:
    fractions, results = experiment2b()
    for result in results:
      accuracies.append(result[0])
      errors.append(result[1])

  plotMachineAccuracy(accuracies, errors, fractions, \
                      'plots/experiment2b_machine_performance')

  plotLabellingAccuracy(labelling_accuracies, fractions, \
                        'plots/experiment2b_labelling_accuracy')

  plotNumberTrainingExamples(number_training_examples, fractions, \
                             'plots/experiment2b_number_examples')

if __name__ == '__main__':
  main()
