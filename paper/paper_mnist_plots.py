import sys
import string
import numpy as np
import matplotlib.pyplot as plt

from keras.optimizers import SGD

sys.path.insert(0,'/Users/dwright/dev/zoo/machine_augmented_classification/experiments/dissolving')
from dissolving_utils import get_cluster_centres

from paper_snhunters_plots import ReDEC
from sklearn.decomposition import PCA

sys.path.insert(0,'/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras')
from datasets import load_mnist
from DEC import DEC, ClusteringLayer, cluster_acc

lcolours = ['#CAA8F5', '#D6FF79', '#A09BE7', '#5F00BA', '#56CBF9', \
            '#F3C969', '#ED254E', '#B0FF92', '#D9F0FF', '#46351D']

# DEC constants from DEC paper
batch_size = 256
lr         = 0.01
momentum   = 0.9
tol        = 0.001
maxiter    = 10
#update_interval = 140 #perhaps this should be 1 for multitask learning
update_interval = 1 #perhaps this should be 1 for multitask learning
n_clusters = 10 # number of clusters to use
n_classes  = 2  # number of classes

def pca_plot(base_network, x, cluster_centres=None, y=None, labels=[], output_file=None,\
             lcolours=[], ulcolour='#747777', ccolour='#4D6CFA', legend=False):
    
  def onpick(event):
    print('picked')
    print(event.ind)
    #print(y[event.ind[0]])
    dim = int(np.ceil(np.sqrt(len(event.ind))))
    print(dim)
    fig = plt.figure()
    for i in range(len(event.ind)):
      ax = fig.add_subplot(dim,dim,i+1)
      ax.imshow(np.reshape(x[event.ind[i]], (28,28)), cmap='gray_r')
      plt.axis('off')
    plt.show()
  
  pca = PCA(n_components=2)
  x_pca = pca.fit_transform(np.nan_to_num(base_network.predict(x)))
  if cluster_centres is not None:
    c_pca = pca.transform(cluster_centres)
  fig = plt.figure(figsize=(6,6))
  ax = fig.add_subplot(111)
  ax.scatter(x_pca[np.where(y!=-1),0], x_pca[np.where(y!=-1),1], marker='o', alpha=0, picker=5)
  if np.any(y):
    unique_targets = list(np.unique(y))
    if -1 in unique_targets:
      ax.scatter(x_pca[np.where(y==-1),0], x_pca[np.where(y==-1),1], marker='o', s=15, \
        color=ulcolour, alpha=0.3)
      unique_targets.remove(-1)
    for l in unique_targets:
        l = int(l)
        ax.scatter(x_pca[np.where(y==l),0], x_pca[np.where(y==l),1], marker='o', s=5, \
          color=lcolours[l], alpha=1.0, label=labels[l])
  else:
    ax.scatter(x_pca[:,0], x_pca[:,1], marker='o', s=15, \
      color=ulcolour, alpha=0.1)
  if cluster_centres is not None:
    ax.scatter(c_pca[:,0], c_pca[:,1], marker='o', s=20, color=ccolour, \
      alpha=1.0, label='cluster centre')

    for i,c in enumerate(string.ascii_lowercase[:len(cluster_centres)]):
      ax.text(c_pca[i,0], c_pca[i,1], str(c), size=21, color='k', weight='bold')
      ax.text(c_pca[i,0], c_pca[i,1], str(c), size=20, color='w')
  plt.axis('off')
  if legend:
    plt.legend(ncol=1,loc='upper left')
  if output_file:
    plt.savefig(output_file)
  fig.canvas.mpl_connect('pick_event', onpick)
  plt.show()

def clickable_analysis(x_test, y_test):
  ae_weights  = '/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras/results/mnist/ae_weights.h5'
  redec = ReDEC(dims=[x_test.shape[-1], 500, 500, 2000, 10], \
                n_clusters=n_clusters, batch_size=batch_size)
  redec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
                        ae_weights=ae_weights,
                        x=x_test)
  redec.model.load_weights('/Users/dwright/dev/zoo/machine_augmented_classification/notebooks/mnist/test/DEC_model_final.h5',by_name=True)
  #pca_plot(redec.encoder, np.concatenate((x_train, x_train_dev, x_valid)), \
  #       y=np.concatenate((y_train_vote_fractions, y_train_dev_vote_fractions, -1*np.ones(y_valid.shape))), \
  #       cluster_centres=get_cluster_centres(redec), labels=['bogus', 'real'], lcolours=['#D138BF','#7494EA'], \
  #       ulcolour='#A0A4B8', ccolour='#21D19F', legend=False)

  pca_plot(redec.encoder, x_test, y=y_test, \
         cluster_centres=get_cluster_centres(redec), labels=[str(i) for i in range(10)], lcolours=lcolours, \
         ulcolour='#A0A4B8', ccolour='#21D19F', legend=False)

def main():
  x, y = load_mnist()
  # split the data into training, validation and test sets
  m = x.shape[0]
  m = m - 20000
  sample_frac = 0.01 # sampling 1% of the points
  split = int(sample_frac*m)
  print(split)

  # the training set acts as the sample of data for which we query volunteer classifications.
  # Here the data is sampled uniformly at random from the entire data set, targeting the most densely populated regions
  # of feature space.
  x_train = x[:split]
  y_train = y[:split]
  x_train_dev = x[split:2*split]
  y_train_dev = y[split:2*split]

  x_valid = x[50000:60000]
  y_valid = y[50000:60000]

  x_test  = x[60000:]
  y_test  = y[60000:]
  print(x_train.shape)

  clickable_analysis(x_test, y_test)

if __name__ == '__main__':
  main()
