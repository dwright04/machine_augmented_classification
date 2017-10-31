import numpy as np

np.random.seed(1337)

from utils import loadMNIST, clusterData, calculateAccuracy
from utils import buildModel, runTrials, getMACResultsDB
from utils import calculateClassDistribution, calculateLabellingAccuracy

from keras.utils import np_utils

def experiment2aLabelleingMethod(X, y, clustering, n_classes):
  
  n_clusters = clustering.n_clusters
  
  for cluster in range(n_clusters):
    cluster_indices = np.where(clustering.labels_ == cluster)[0]
    n_assigned_examples = cluster_indices.shape[0]
    cluster_labels = y[cluster_indices]
    cluster_label_fractions = np.mean(cluster_labels, axis=0)
    dominant_cluster_class = np.argmax(cluster_label_fractions)
    print(cluster+1, n_assigned_examples, dominant_cluster_class, \
          cluster_label_fractions[dominant_cluster_class])
    """
    print('%d & %d & %d & %.3lf & %.3lf & %.3lf & %.3lf & %.3lf &' \
          ' %.3lf & %.3lf & %.3lf & %.3lf & %.3lf\\\\' % \
      (cluster+1, n_assigned_examples, dominant_cluster_class, \
       cluster_label_fractions[0], cluster_label_fractions[1], \
       cluster_label_fractions[2], cluster_label_fractions[3], \
       cluster_label_fractions[4], cluster_label_fractions[5], \
       cluster_label_fractions[6], cluster_label_fractions[7], \
       cluster_label_fractions[8], cluster_label_fractions[9]))
    """
    # assign labels based on dominant class
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

def experiment2a():

  image_dim = 28
  n_classes = 10
  n_trials = 5
  epochs = 20
  batch_size = 500
  
  db = getMACResultsDB()
  collection = db['experiment2a']

  x_train, y_train, x_train_flattened, x_test, y_test = loadMNIST()

  # calculate performance on gold labels
  y_train = np_utils.to_categorical(y_train, n_classes)
  y_test = np_utils.to_categorical(y_test, n_classes)
  
  gold_benchmark = \
    runTrials(x_train, y_train, x_test, y_test, n_trials, n_classes, \
              data='mnist_gold', epochs=epochs)

  print(gold_benchmark)

  doc = {
          'name': 'gold benchmark',
          'm': y_train.shape[0],
          'accuracy': gold_benchmark[0],
          'error': gold_benchmark[1],
          'trials accuracy': gold_benchmark[2],
          'labelling accuracy': calculateLabellingAccuracy(y_train, y_train),
          'training set class distribution': \
            calculateClassDistribution(y_train).tolist()
        }

  collection.insert_one(doc)

  clustering = clusterData(x_train_flattened)
  
  x_labelled, labels, labelled_indices, unlabelled_indices = \
   experiment2aLabelleingMethod(x_train, y_train, \
                                clustering, n_classes)

  result = \
    runTrials(x_labelled, labels, x_test, y_test, n_trials, n_classes, \
              data='mnist_experiment2a', epochs=epochs)

  print(result)

  print(calculateLabellingAccuracy(y_train[labelled_indices], labels))
  doc = {
          'name': 'majority class label assignment',
          'm': labels.shape[0],
          'accuracy': result[0],
          'error': result[1],
          'trials accuracy': result[2],
          'labelling accuracy': \
            calculateLabellingAccuracy(y_train[labelled_indices], labels),
          'training set class distribution': \
            calculateClassDistribution(labels).tolist()
        }

  collection.insert_one(doc)

def main():
  db = getMACResultsDB()
  experiment_name = 'experiment2a'
  try:
    assert experiment_name in db.collection_names()
    cursor = db[experiment_name].find()
    for doc in cursor:
      print(doc)
  except AssertionError:
    experiment2a()

if __name__ == '__main__':
  main()
