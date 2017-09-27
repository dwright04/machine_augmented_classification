import sys
sys.path.insert(0,'/Users/dwright/dev/zoo/machine_augmented_classification/' +\
  'experiments/experiment2')

import numpy as np

np.random.seed(1337)

from utils import loadThreePi, clusterData, calculateAccuracy
from utils import buildModel, runTrials, getMACResultsDB
from utils import calculateClassDistribution, calculateLabellingAccuracy

from keras.utils import np_utils

from experiment2a import experiment2aLabelleingMethod

def experiment5a():

  image_dim = 20
  n_classes = 2
  n_trials = 5
  epochs = 20
  batch_size = 500
  
  db = getMACResultsDB()
  collection = db['experiment5a']
  
  x_train, y_train, x_train_flattened, x_test, y_test = loadThreePi()

  # calculate performance on gold labels
  y_train = np_utils.to_categorical(y_train, n_classes)
  y_test = np_utils.to_categorical(y_test, n_classes)
  
  gold_benchmark = \
    runTrials(x_train, y_train, x_test, y_test, n_trials, n_classes, \
              data='3pi_gold', epochs=epochs, image_dim=image_dim)

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

  clustering = clusterData(x_train_flattened, data='3pi')
  
  x_labelled, labels, labelled_indices, unlabelled_indices = \
   experiment2aLabelleingMethod(x_train, y_train, \
                                clustering, n_classes)

  result = \
    runTrials(x_labelled, labels, x_test, y_test, n_trials, n_classes, \
              data='3pi_experiment5a', epochs=epochs, image_dim=image_dim)

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
  collection_name = 'experiment5a'
  experiment5a()
  try:
    assert collection_name in db.collection_names()
    cursor = db[collection_name].find()
    for doc in cursor:
      print(doc)
  except AssertionError:
    experiment5a()

if __name__ == '__main__':
  main()
