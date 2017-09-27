import sys
sys.path.insert(0,'/Users/dwright/dev/zoo/machine_augmented_classification/' +\
  'experiments/experiment2')

import numpy as np

np.random.seed(1337)

from utils import loadThreePi, clusterData, calculateAccuracy
from utils import buildModel, runTrials, getMACResultsDB
from utils import calculateClassDistribution, calculateLabellingAccuracy

from keras.utils import np_utils

from experiment2b import experiment2bLabelleingMethod, plotMachineAccuracy
from experiment2b import plotLabellingAccuracy, plotNumberTrainingExamples

def experiment5b():

  image_dim = 20
  n_classes = 2
  n_trials = 5
  epochs = 20
  batch_size = 500

  db = getMACResultsDB()
  collection = db['experiment5b']
  
  x_train, y_train, x_train_flattened, x_test, y_test = loadThreePi()

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
                data='mnist_experiment5b_fraction%.2lf'%(fraction), \
                epochs=epochs, image_dim=image_dim)

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

def main():
  db = getMACResultsDB()
  experiment_name = 'experiment5b'
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
    fractions, results = experiment5b()
    for result in results:
      accuracies.append(result[0])
      errors.append(result[1])

  plotMachineAccuracy(accuracies, errors, fractions, \
                      'plots/experiment5b_machine_performance', \
                      'experiment5a')

  plotLabellingAccuracy(labelling_accuracies, fractions, \
                        'plots/experiment5b_labelling_accuracy', \
                        'experiment5a')

  plotNumberTrainingExamples(number_training_examples, fractions, \
                             'plots/experiment5b_number_examples', \
                             'experiment5a')

if __name__ == '__main__':
  main()
