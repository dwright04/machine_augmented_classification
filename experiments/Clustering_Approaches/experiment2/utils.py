import pickle
import numpy as np
import scipy.io as sio

from sklearn.cluster import AgglomerativeClustering

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

from pymongo import MongoClient

# https://coolors.co/b8336a-726da8-7d8cc4-a0d2db-c490d1

def loadMNIST(m=10000):
 
  # Load pre-shuffled MNIST data into train and test sets
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # flatten the images for PCA
  x_train_flattened = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] \
                                                           * x_train.shape[2]))

  # limit the number of examples to 10000 so we can work with plotly
  # interactive plots
  x_train = x_train[:m][:,:,:,np.newaxis]
  x_train_flattened = x_train_flattened[:m]
  y_train = y_train[:m]

  return x_train, y_train, x_train_flattened, x_test[:,:,:,np.newaxis], y_test

def loadThreePi():
  path = '/Users/dwright/dev/zoo/data/'
  file = '3pi_20x20_skew2_signPreserveNorm.mat'
  data = sio.loadmat(path+file)
  x_train_flattened = data['X'] # load the pixel data
  m = x_train_flattened.shape[0]
  x_train = np.reshape(x_train_flattened, (m,20,20,1), order='F')
  y_train = np.squeeze(data['y']) # load the targets
  x_test  = data['testX'] # load the pixel data
  x_test = np.reshape(x_test, (x_test.shape[0],20,20,1), order='F')
  y_test  = np.squeeze(data['testy']) # load the targets
  return  x_train, y_train, x_train_flattened, x_test, y_test

def clusterData(X, data='mnist', n_clusters=100):
  filename = 'data/%s_clustering_%d.pkl'%(data, n_clusters)
  try:
    clustering = pickle.load(open(filename,'rb'))
    print('[*] Precomputed clustering loaded.')
  except IOError:
    print('[*] Clustering...')
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clustering.fit(X)
    pickle.dump(clustering, open(filename,'wb'))
  return clustering

def calculateAccuracy(model, X, y, n_classes):
  pred = model.predict(X)
  return calculateLabellingAccuracy(y, pred)

def calculateLabellingAccuracy(y, pred):
  return 100*np.sum(np.argmax(pred, axis=1) \
         == np.argmax(y, axis=1)) \
         /  float(y.shape[0])

def buildModel(image_dim, n_classes):
  model = Sequential()
  model.add(Conv2D(filters=16, kernel_size=2, padding='valid', \
                   activation='relu', input_shape=(image_dim,image_dim,1)))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Conv2D(filters=32, kernel_size=2, padding='valid', \
                   activation='relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Conv2D(filters=64, kernel_size=2, padding='valid', \
                   activation='relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(GlobalAveragePooling2D('channels_last'))
  model.add(Dropout(0.3))
  model.add(Dense(n_classes, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  return model

def runTrials(x_train, y_train, x_test, y_test, n_trials, n_classes, \
  data='mnist', image_dim=28, epochs=20, batch_size=500):
  
  accs = []
  for trial in range(1,n_trials+1):
    filename='data/%s_model_trial%d_epochs%d_batchsize%d.h5' % \
      (data,trial,epochs,batch_size)
    try:
      model = load_model(filename)
    except IOError:
      model = buildModel(image_dim, n_classes)
      model.fit(x_train, y_train, epochs=epochs, \
                batch_size=batch_size, verbose=1)
      model.save(filename)

    accs.append(calculateAccuracy(model, x_test, \
                                  y_test, n_classes))

  return np.mean(accs), np.std(accs), accs

def calculateClassDistribution(y):
  return np.sum(y, axis=0)/np.sum(y)

def getMACResultsDB():

  client = MongoClient()
  db = client.MACResults
  return db
