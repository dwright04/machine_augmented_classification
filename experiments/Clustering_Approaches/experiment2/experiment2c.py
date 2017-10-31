import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)

from utils import loadMNIST, clusterData, calculateAccuracy
from utils import buildModel, runTrials, getMACResultsDB
from utils import calculateClassDistribution, calculateLabellingAccuracy

from experiment2b import plotMachineAccuracy, plotLabellingAccuracy
from keras.utils import np_utils

def experiment2cLabelleingMethod(X, y, clustering, n_classes, fraction):

  n_clusters = clustering.n_clusters
  sampled_dominant_cluster_classes = []
  dominant_cluster_classes = []
  for cluster in range(n_clusters):
    cluster_indices = np.where(clustering.labels_ == cluster)[0]
    n_assigned_examples = cluster_indices.shape[0]
    # what number of subjects this fraction produces
    n_queried_examples = int(np.ceil(fraction*n_assigned_examples))
    sampled_cluster_indices = cluster_indices[:n_queried_examples]
    # select a subset
    sampled_cluster_labels = y[sampled_cluster_indices]
    cluster_labels = y[cluster_indices]
    sampled_cluster_label_fractions = np.mean(sampled_cluster_labels, axis=0)
    cluster_label_fractions = np.mean(cluster_labels, axis=0)
    sampled_dominant_cluster_class = \
      np.argmax(sampled_cluster_label_fractions)
    dominant_cluster_class = np.argmax(cluster_label_fractions)

    sampled_dominant_cluster_classes.append(sampled_dominant_cluster_class)
    dominant_cluster_classes.append(dominant_cluster_class)
    x = X[cluster_indices]
  
    l = np.zeros((x.shape[0], 10))
    l[:,sampled_dominant_cluster_class] += 1
    # build the training set labelling subjects with the dominant cluster
    # class the sampling produced
    try:
      x_labelled = np.concatenate((x_labelled, x))
      labels = np.concatenate((labels, l))
      labelled_indices = np.concatenate((labelled_indices, cluster_indices))
    except (NameError, ValueError):
      x_labelled = x
      labels = l
      labelled_indices = cluster_indices
    
  m = x_labelled.shape[0]
  order = np.random.permutation(m)
  x_labelled = x_labelled[order]
  labels = labels[order]
  labelled_indices = labelled_indices[order]
  
  unlabelled_indices = np.array([x for x in range(X.shape[0]) \
                                 if x not in labelled_indices])
                                 
  return x_labelled, labels, labelled_indices, unlabelled_indices, \
    sampled_dominant_cluster_classes, dominant_cluster_classes

def experiment2c():
  image_dim = 28
  n_classes = 10
  n_trials = 5
  epochs = 20
  batch_size = 500
  
  db = getMACResultsDB()
  collection = db['experiment2c']

  x_train, y_train, x_train_flattened, x_test, y_test = loadMNIST()

  y_train = np_utils.to_categorical(y_train, n_classes)
  y_test = np_utils.to_categorical(y_test, n_classes)

  clustering = clusterData(x_train_flattened)

  fractions = [0.00, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

  recovery = []
  results = []
  for fraction in fractions:
    x_labelled, labels, labelled_indices, unlabelled_indices, \
    sampled_dominant_cluster_classes, dominant_cluster_classes = \
      experiment2cLabelleingMethod(x_train, y_train, clustering,
                                   n_classes, fraction)

    result = \
      runTrials(x_labelled, labels, x_test, y_test, n_trials, n_classes, \
                data='mnist_experiment2c_fraction%.2lf'%(fraction), \
                epochs=epochs)
              
    results.append(result)
    
    recovery.append(calculateLabellingAccuracy(
      np_utils.to_categorical(dominant_cluster_classes),
      np_utils.to_categorical(sampled_dominant_cluster_classes))/100.0)
      
    doc = {
          'name': 'majority class recovery by cluster subsample',
          'm': labels.shape[0],
          'fraction sampled': fraction,
          'accuracy': result[0],
          'error': result[1],
          'trials accuracy': result[2],
          'labelling accuracy': \
            calculateLabellingAccuracy(y_train[labelled_indices], labels),
          'training set class distribution': \
            calculateClassDistribution(labels).tolist(),
          'cluster majority class recovery rate': calculateLabellingAccuracy(
            np_utils.to_categorical(dominant_cluster_classes),
            np_utils.to_categorical(sampled_dominant_cluster_classes))/100.0
        }
        
    collection.insert_one(doc)

  return fractions, results, recovery

def plotMajorityClassRecovery(fractions, recovery, filename):
  fig, ax = plt.subplots()
  ax.plot(fractions, recovery, 'o', color='#7D8CC4')
  ax.plot(np.arange(-0.02,1.03, 0.01), \
    np.ones((np.arange(-0.02,1.03, 0.01).shape)), 'k--')
  ax.set_xlabel('fraction of subjects sampled from cluster')
  ax.set_ylabel('correct cluster majority class recovery rate')
  ax.set_xlim(-0.02,1.03)
  #plt.show()
  plt.savefig(filename+'.pdf')
  plt.savefig(filename+'.png')

def main():
  db = getMACResultsDB()
  experiment_name = 'experiment2c'
  accuracies = []
  labelling_accuracies = []
  recoveries = []
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
      fractions.append(doc['fraction sampled'])
      recoveries.append(doc['cluster majority class recovery rate'])
  except AssertionError:
    fractions, results, recovery = experiment2c()
    for i,result in enumerate(results):
      accuracies.append(result[0])
      errors.append(result[1])
      recoveries.append(recovery[i])

  plotMachineAccuracy(accuracies, errors, fractions, \
                      'plots/experiment2c_machine_performance')

  plotLabellingAccuracy(labelling_accuracies, fractions, \
                        'plots/experiment2c_labelling_accuracy')

  #fractions, recovery = experiment2c()
  plotMajorityClassRecovery(fractions, recoveries, \
                            'plots/experiment2c_majority_class_recovery')

if __name__ == '__main__':
  main()
