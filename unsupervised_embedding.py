import numpy as np

from numpy import linalg as LA

from sklearn.cluster import KMeans

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers

class UnsupervisedEmbedding(object):
  def __init__(self, units, clusters, epochs, learning_rate=0.01, alpha=1.):
    self.units = units
    self.clusters = clusters
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.alpha = alpha
    
    self.cluster_centres = None
    self.weights = {}

  def soft_assignment(self, X, alpha=1.):
    m = X.shape[0]
    q = np.ones((m, self.clusters))
    for j in range(self.clusters):
      a = np.power((1+self.kmeans.transform(X)[:,j])/alpha, -(alpha+1)/2.)
      b = 0
      for k in range(self.clusters):
        if k != j:
          b += np.power((1+self.kmeans.transform(X)[:,k])/alpha, -(alpha+1)/2.)
      q[:,j] *= a/b
    return q
  
  def get_target_distribution(self, q):
    p = np.ones(q.shape)
    for j in range(self.clusters):
      a = (q[:,j] * q[:,j]) / np.sum(q[:,j])
      b = 0
      for k in range(self.clusters):
        if k != j:
          b += (q[:,k] * q[:,k]) / np.sum(q[:,k])
      p[:,j] *= a/b
    return p

  def KL_divergence(self, q, p):
    return np.sum(p*np.log(p/q))

  def get_d_loss_z(self, X_encoded, q, p, alpha=1.):
    d_loss_z = 0
    for j in range(self.clusters):
      d_loss_z += np.tile(1/(1+self.kmeans.transform(X_encoded)[:,j]/alpha), (self.units,1)).T \
                   * np.tile(p[:,j]-q[:,j], (self.units,1)).T \
                   * (X_encoded-np.tile(self.cluster_centres[j,:], (X_encoded.shape[0],1)))
    
    d_loss_z = (alpha+1)/alpha * d_loss_z
    
    return d_loss_z

  def get_d_loss_mu(self, X_encoded, q, p, alpha=1.):
    d_loss_mu = 0
    for i  in range(X_encoded.shape[0]):
      d_loss_mu += np.tile(1/(1+self.kmeans.transform(X_encoded)[i,:]/alpha), (self.units,1)).T \
                 * np.tile(p[i,:]-q[i,:], (self.units,1)).T \
                 * (X_encoded[i,:]-self.cluster_centres)
    d_loss_mu = -(alpha + 1)/ alpha * d_loss_mu
    
    return d_loss_mu

  def compute_gradient(self, X, q, p):
    d_loss_z = self.get_d_loss_z(X, q, p)
    d_loss_mu = self.get_d_loss_mu(X, q, p)
      
    gradW = np.zeros(self.weights[1]['W'].shape)
    gradb = np.zeros(self.weights[1]['b'].shape)
    # TODO: Use batch_size
    for i in range(X.shape[0]):
      
      self.cluster_centres = self.cluster_centres \
                         - self.learning_rate \
                         * self.get_d_loss_mu(X[i][np.newaxis], q, p, self.alpha)
        
      delta_out = d_loss_z[i]

      gradW += np.dot(delta_out,X[i].T)
      gradb += delta_out

    return gradW, gradb
  
  def fit(self, X, y, batch_szie=256):
    m = float(X.shape[0])
    self.initialise(X, y)
    losses = []
    for i in range(self.epochs):
      X_encoded = self.encode(X)
      q = self.soft_assignment(X_encoded)
      p = self.get_target_distribution(q)
      
      order = np.random.permutation(q.shape[0])
      
      X_encoded = X_encoded[order]
      q = q[order]
      p = p[order]
      
      loss = self.KL_divergence(q, p)
      print(loss)
      
      losses.append(loss)
      
      gradW, gradb = self.compute_gradient(X_encoded, q, p)
        
      self.weights[1]['W'] = 1/m * self.weights[1]['W'] \
                           - self.learning_rate \
                           * gradW
                             
      self.weights[1]['b'] = 1/m * self.weights[1]['b'] \
                           - self.learning_rate \
                           * gradb

    plt.plot(range(self.epochs), losses)
    plt.show()
  
  def initialise(self, X, y):
    AE1 = Sequential()
    AE1.add(Dropout(0.2, input_shape=(X.shape[1],)))
    AE1.add(Dense(self.units, activation='relu'))
    AE1.add(Dropout(0.2))
    AE1.add(Dense(X.shape[1], activation='sigmoid'))
    AE1.compile(loss='binary_crossentropy', optimizer='rmsprop')
    AE1.fit(X, y, epochs=10, batch_size=256)

    self.weights[1] = {'W': AE1.layers[1].get_weights()[0],
                       'b': AE1.layers[1].get_weights()[1]}
    
    self.weights[2] = {'W': AE1.layers[3].get_weights()[0],
                       'b': AE1.layers[3].get_weights()[1]}
    
    X_encoded = self.encode(X)
    
    self.kmeans = KMeans(n_clusters=self.clusters).fit(X_encoded)
    
    self.cluster_centres = self.kmeans.cluster_centers_

  def relu(self, X):
    return np.maximum(X, 0, X)
  
  def d_relu(self, X, epsilon=1e-9):
    return X + epsilon > 0

  def encode(self, X):
    z1 = np.dot(X,self.weights[1]['W']) + self.weights[1]['b']
    return self.relu(z1)

  def computeNumericalGradient(self, X):
    
    tmp = self.weights.copy()
    b_shape = self.weights[1]['b'].shape
    W_shape = self.weights[1]['W'].shape
    params = np.concatenate((np.ravel(self.weights[1]['b']), np.ravel(self.weights[1]['W'])))
    numgrad = np.zeros(np.shape(params))
    perturb = np.zeros(np.shape(params))
    epsilon = 0.0001
    for i in range(len(params)):
      perturb[i] = epsilon
      self.weights[1]['b'] = np.reshape((params-perturb)[:b_shape[0]], b_shape)
      self.weights[1]['W'] = np.reshape((params-perturb)[b_shape[0]:], W_shape)
      X_encoded = self.encode(X)
      q = self.soft_assignment(X_encoded)
      p = self.get_target_distribution(q)
      loss1 = self.KL_divergence(q,p)
      self.weights[1]['b'] = np.reshape((params+perturb)[:b_shape[0]], b_shape)
      self.weights[1]['W'] = np.reshape((params+perturb)[b_shape[0]:], W_shape)
      X_encoded = self.encode(X)
      q = self.soft_assignment(X_encoded)
      p = self.get_target_distribution(q)
      loss2 = self.KL_divergence(q,p)
      numgrad[i] = (loss2 - loss1) / (2.0 * epsilon)
      perturb[i] = 0
    self.weights = tmp
    return numgrad

  def check_gradients(self, X):
    self.initialise(X, X)
    X_encoded = self.encode(X)
    
    q = self.soft_assignment(X_encoded)
    p = self.get_target_distribution(q)
    gradW, gradb = self.compute_gradient(X_encoded, q, p)

    grad = np.concatenate((np.ravel(gradb), np.ravel(gradW)))
    numgrad = self.computeNumericalGradient(X)
    for i in range(len(numgrad)):
      print "%d\t%f\t%f" % (i, numgrad[i], grad[i])
    diff = numgrad-grad
    print diff

def main():

  from keras.datasets import mnist
  # Load pre-shuffled MNIST data into train and test sets
  (x_train, _), (x_test, _) = mnist.load_data()
  # flatten the images for PCA
  x_train_flattened = np.reshape(x_train, \
    (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
  x_train_flattened = x_train_flattened[:1000, :10] / 255.
  
  ue = UnsupervisedEmbedding(units=6, clusters=4, epochs=100)
  #ue.fit(x_train_flattened, x_train_flattened)

  ue.check_gradients(x_train_flattened)

if __name__ == '__main__':
  main()
