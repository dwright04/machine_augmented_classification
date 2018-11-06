import sys
import pickle
import scipy.io as sio
import numpy as np
import keras.backend as K
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from sklearn.metrics import f1_score, roc_curve, homogeneity_score, normalized_mutual_info_score
from sklearn.cluster import KMeans

sys.path.insert(0,'/Users/dwright/dev/zoo/machine_augmented_classification/experiments/dissolving/mutitask/snhunters')
from multitask_experiment import MultitaskDEC

sys.path.insert(0,'/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras')
from DEC import DEC

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

def purity_score(n_assigned_list, majority_class_fractions):
  m = np.sum(n_assigned_list)
  mask = np.where(n_assigned_list!=0)
  return np.mean((np.array(n_assigned_list)[mask] / m) * np.array(majority_class_fractions)[mask])

def get_cluster_to_label_mapping_safe(y, y_pred, n_classes, n_clusters, toprint=True):
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
    if toprint:
      print(cluster, n_assigned_examples, majority_cluster_class, cluster_label_fractions[majority_cluster_class])
  print(purity_score(n_assigned_list, majority_class_fractions))
  #print(cluster_to_label_mapping)
  if toprint:
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
  if toprint:
    print(cluster_to_label_mapping)
  return cluster_to_label_mapping, n_assigned_list, majority_class_fractions, 0.0

def calc_f1_score(y_true, predicted_clusters, cluster_to_label_mapping):
  y_pred = []
  for i in range(len(y_true)):
    y_pred.append(cluster_to_label_mapping[predicted_clusters[i]])
  return f1_score(y_true, np.array(y_pred))

def load_data_set():
  data = sio.loadmat('/Users/dwright/dev/zoo/machine_augmented_classification/data/snhunters/3pi_20x20_skew2_signPreserveNorm.mat')
  x_test  = data['testX']
  y_test  = np.squeeze(data['testy'])
  x_train  = data['X']
  y_train  = np.squeeze(data['y'])
  m = x_train.shape[0]
  order = np.random.permutation(m)
  x_train = x_train[order]
  y_train = y_train[order]
  split1 = int(.5*m)
  split2 = int(.75*m)
  x_valid = x_train[split1:split2]
  y_valid = y_train[split1:split2]
  x_train_dev = x_train[split2:]
  y_train_dev = y_train[split2:]
  x_train = x_train[:split1]
  y_train = y_train[:split1]
  return x_train, y_train, x_train_dev, y_train_dev, x_valid, y_valid, x_test, y_test, order, split1, split2

def get_labels_first_human():
  data = sio.loadmat('/Users/dwright/dev/zoo/machine_augmented_classification/data/snhunters/3pi_20x20_skew2_signPreserveNorm_volunteer_votes_first_only.mat')
  y_train_first_human = np.squeeze(data['y']) == 'Yes'
  y_test_first_human = np.squeeze(data['y_test']) == 'Yes'

  return y_train_first_human, y_test_first_human

def get_labels_vote_fractions(order, split1, split2):
  data = sio.loadmat('/Users/dwright/dev/zoo/machine_augmented_classification/data/snhunters/3pi_20x20_skew2_signPreserveNorm.mat')
  vote_fraction_dict = pickle.load(open('/Users/dwright/dev/zoo/data/supernova-hunters-vote-fraction-dict.pkl', 'rb'))
  train_files = []
  for f in data['train_files']:
    train_files.append(f.strip().replace('.fits',''))
  test_files = []
  for f in data['test_files']:
    test_files.append(f.strip().replace('.fits',''))
  vote_fractions_train = []
  vote_fractions_test = []
  seen = []
  for diff in train_files:
    vote_fractions_train.append(vote_fraction_dict[diff])
    seen.append(diff)
  for diff in test_files:
    vote_fractions_test.append(vote_fraction_dict[diff])

  vote_fractions_train = np.array(vote_fractions_train)
  vote_fractions_test = np.array(vote_fractions_test)

  vote_fractions_train = vote_fractions_train[order]
  vote_fractions_train_dev = vote_fractions_train[split2:]
  vote_fractions_train = vote_fractions_train[:split1]

  y_train_vote_fractions = vote_fractions_train > 0.5
  y_train_dev_vote_fractions = vote_fractions_train_dev > 0.5
  y_test_vote_fractions = vote_fractions_test > 0.5

  return y_train_vote_fractions, y_train_dev_vote_fractions, y_test_vote_fractions

class ReDEC(DEC):
    def clustering(self, train_data, valid_data=None, test_data=None,
                   tol=1e-3,
                   update_interval=140,
                   maxiter=2e4,
                   pretrained_weights=None,
                   metrics_dict=None,
                   last_ite=0,
                   save_dir='./results/dec'):
        #x = np.concatenate((x_train, x_test))
        x = train_data[0]
        print('Update interval', update_interval)
        save_interval = x.shape[0] / self.batch_size * 5  # 5 epochs
        print('Save interval', save_interval)

        frame_index = 1

        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if metrics_dict is None:
          metrics_dict = {'iteration':[],
                          'train_f1c':[],
                          'train_h':[],
                          'train_nmi':[],
                          'valid_f1c':[],
                          'valid_h':[],
                          'valid_nmi':[]
                          }
        loss = 0
        index = 0
        for ite in range(int(maxiter)):
            if self.video_path:
                self.model.save_weights(self.video_path+'/%s_%06d_weights.h5'%('clustering', frame_index))
                frame_index += 1

            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if ite > 0:
                  delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if valid_data is not None:
                  loss = np.round(loss, 5)
                  train_q = self.model.predict(train_data[0], verbose=0)
                  train_p = self.target_distribution(train_q)
                  c_map, _, _, train_ps = \
                    get_cluster_to_label_mapping_safe(train_data[1], train_q.argmax(1), n_classes, n_clusters, toprint=False)
                  train_cluster_pred = self.model.predict(train_data[0], verbose=0).argmax(1)
                  f1c = np.round(calc_f1_score(train_data[1], train_cluster_pred, c_map), 5)
                  h = np.round(homogeneity_score(train_data[1], train_cluster_pred), 5)
                  nmi = np.round(normalized_mutual_info_score(train_data[1], train_cluster_pred), 5)
                  train_ps = np.round(train_ps, 5)
                
                  valid_q = self.model.predict(valid_data[0], verbose=0)
                  valid_p = self.target_distribution(valid_q)
                  val_loss = np.round(self.model.test_on_batch(valid_data[0], valid_p), 5)
                  valid_cluster_pred = valid_q.argmax(1)
                  f1c_valid = np.round(calc_f1_score(valid_data[1], valid_cluster_pred, c_map), 5)
                  h_valid = np.round(homogeneity_score(valid_data[1], valid_cluster_pred), 5)
                  nmi_valid = np.round(normalized_mutual_info_score(valid_data[1], valid_cluster_pred), 5)
                  _, _, _, valid_ps = \
                    get_cluster_to_label_mapping_safe(valid_data[1], valid_cluster_pred, n_classes, n_clusters, toprint=False)
                  valid_ps = np.round(valid_ps, 5)
                
                  print('Iter', ite,', F1c=', f1c, 'h=', h, 'nmi=', nmi, 'ps=', train_ps, '; loss=', loss, ';', \
                        'vF1c=', f1c_valid,'vh=', h_valid, 'vnmi=', nmi_valid, 'v ps=',valid_ps, '; vloss=,', val_loss)
                  metrics_dict['iteration'].append(last_ite+ite)
                  metrics_dict['train_f1c'].append(f1c)
                  metrics_dict['train_h'].append(h)
                  metrics_dict['train_nmi'].append(nmi)
                  metrics_dict['train_ps'].append(train_ps)
                  metrics_dict['valid_f1c'].append(f1c_valid)
                  metrics_dict['valid_h'].append(h_valid)
                  metrics_dict['valid_nmi'].append(nmi_valid)
                  metrics_dict['valid_ps'].append(valid_ps)
                # check stop criterion
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # train on batch
            if (index + 1) * self.batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * self.batch_size::],
                                                 y=p[index * self.batch_size::])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * self.batch_size:(index + 1) * self.batch_size],
                                                 y=p[index * self.batch_size:(index + 1) * self.batch_size])
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save IDEC model checkpoints
                print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        print('saving model to:', save_dir + '/DEC_model_final.h5')
        self.model.save_weights(save_dir + '/DEC_model_final.h5')
        if test_data:
          y_pred_test = self.model.predict(test_data[0])
          test_cluster_pred = self.model.predict(test_data[0], verbose=0).argmax(1)
          f1c_test = np.round(calc_f1_score(test_data[1], test_cluster_pred, c_map), 5)
          h_test = np.round(homogeneity_score(test_data[1], test_cluster_pred), 5)
          nmi_test = np.round(normalized_mutual_info_score(test_data[1], test_cluster_pred), 5)
          _, _, _, test_ps = \
            get_cluster_to_label_mapping_safe(test_data[1], test_cluster_pred, n_classes, n_clusters, toprint=False)
          test_ps = np.round(test_ps, 5)
          print('tF1c=', f1c_test,'th=', h_test, 'tnmi=', nmi_test, 'tps=', test_ps)
        return y_pred, metrics_dict

def get_initial_results(x_train,y_train,x_test,y_test):
  ae_weights  = '/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras/results/snhunters/train_only/ae_weights.h5'
  dec_weights = '/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras/results/snhunters/train_only/10/DEC_model_final.h5'
  dec = DEC(dims=[x_train.shape[-1], 500, 500, 2000, 10], \
                   n_clusters=n_clusters, batch_size=batch_size)
  dec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
                       ae_weights=ae_weights,
                       x=x_train)
  dec.model.load_weights(dec_weights)

  c_map, _, _ = \
    get_cluster_to_label_mapping_safe(y_train, dec.predict_clusters(x_train), n_classes, n_clusters, toprint=True)
  test_cluster_pred = dec.predict_clusters(x_test)
  print(np.round(calc_f1_score(y_test, test_cluster_pred, c_map), 5))
  print(np.round(homogeneity_score(y_test, test_cluster_pred), 5))
  print(np.round(normalized_mutual_info_score(y_test, test_cluster_pred), 5))
  #print(np.round(get_cluster_to_label_mapping_safe(y_test, test_cluster_pred, n_classes, n_clusters), 5))

  c_map, _, _ = \
    get_cluster_to_label_mapping_safe(y_test, dec.predict_clusters(x_test), n_classes, n_clusters, toprint=True)
  test_cluster_pred = dec.predict_clusters(x_test)
  print(np.round(calc_f1_score(y_test, test_cluster_pred, c_map), 5))
  print(np.round(homogeneity_score(y_test, test_cluster_pred), 5))
  print(np.round(normalized_mutual_info_score(y_test, test_cluster_pred), 5))
  #print(np.round(get_cluster_to_label_mapping_safe(y_test, test_cluster_pred, n_classes, n_clusters), 5))

def bench_marks(x_train, y_train, x_test, y_test, y_test_first_human, y_train_vote_fractions, y_test_vote_fractions):
  data = sio.loadmat('/Users/dwright/dev/zoo/machine_augmented_classification/data/snhunters/3pi_20x20_skew2_signPreserveNorm_volunteer_votes_first_only.mat')
  first_human_f1_benchmark = f1_score(y_test, y_test_first_human)
  vote_fraction_f1_benchmark = f1_score(y_test, y_test_vote_fractions)
  threepi_cnn_test_f1_benchmark = 0.924951892239
  # expert benchmark from my thesis finding 10 misclassified images in test set.
  y_test_cleaned = y_test.copy()
  y_test_cleaned[np.where(y_test==1)[0][:10]] = y_test_cleaned[np.where(y_test==1)[0][:10]] - 1
  human_expert_f1_benchmark = f1_score(y_test_cleaned, y_test)
  #print(human_expert_f1_benchmark)
  all_ones_f1_benchmark = f1_score(y_test, np.ones(y_test.shape))
  
  
  # load the pretrained DEC model for Supernova Hunters
  ae_weights  = '/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras/results/snhunters/train_only/ae_weights.h5'
  dec_weights = '/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras/results/snhunters/train_only/10/DEC_model_final.h5'
  dec = MultitaskDEC(dims=[x_train.shape[-1], 500, 500, 2000, 10], \
                   n_clusters=n_clusters, batch_size=batch_size)
  dec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
                       ae_weights=ae_weights,
                       x=x_test)
  dec.model.load_weights(dec_weights)
  train_q = dec.model.predict(x_train, verbose=0)
  c_map, _, _ = \
    get_cluster_to_label_mapping_safe(y_train_vote_fractions, train_q.argmax(1), n_classes, n_clusters, toprint=True)
  test_q = dec.model.predict(x_test, verbose=0)
  test_p = dec.target_distribution(test_q)
  test_cluster_pred = test_q.argmax(1)
  unsupervised_f1c_benchmark = np.round(calc_f1_score(y_test, test_cluster_pred, c_map), 5)

  c_map, _, _ = \
    get_cluster_to_label_mapping_safe(y_test, test_q.argmax(1), n_classes, n_clusters, toprint=True)
  test_q = dec.model.predict(x_test, verbose=0)
  test_p = dec.target_distribution(test_q)
  test_cluster_pred = test_q.argmax(1)
  f1c_test = np.round(calc_f1_score(y_test, test_cluster_pred, c_map), 5)
  h_test = np.round(homogeneity_score(y_test, test_cluster_pred), 5)
  nmi_test = np.round(normalized_mutual_info_score(y_test, test_cluster_pred), 5)
  print(f1c_test, h_test, nmi_test)
  return all_ones_f1_benchmark, unsupervised_f1c_benchmark, first_human_f1_benchmark, \
    vote_fraction_f1_benchmark, threepi_cnn_test_f1_benchmark, human_expert_f1_benchmark

def load_metrics_dict(metrics_dict_file):
  return pickle.load(open(metrics_dict_file, 'rb'))

def get_multitask_step_results(x_train, y_train_vote_fractions, x_test, y_test):
  ae_weights  = '/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras/results/snhunters/train_only/ae_weights.h5'
  #dec_weights = '/Users/dwright/dev/zoo/machine_augmented_classification/notebooks/best_train_dev_loss_2.hf'
  dec_weights = '/Users/dwright/dev/zoo/machine_augmented_classification/notebooks/results/dec/DEC_model_0_4.h5'
  dec = MultitaskDEC(dims=[x_train.shape[-1], 500, 500, 2000, 10], \
                      n_clusters=n_clusters, batch_size=batch_size)
  dec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
                        ae_weights=ae_weights,
                        x=x_train)
  dec.model.load_weights(dec_weights, by_name=True)
  train_q = dec.model.predict(x_train, verbose=0)
  c_map, _, _ = \
    get_cluster_to_label_mapping_safe(y_train_vote_fractions, train_q.argmax(1), n_classes, n_clusters, toprint=True)
  test_q = dec.model.predict(x_test, verbose=0)
  test_p = dec.target_distribution(test_q)
  test_cluster_pred = test_q.argmax(1)
  print(np.round(calc_f1_score(y_test, test_cluster_pred, c_map), 5))

  y_pred, metrics_dict, best_ite = dec.clustering(x_train, np_utils.to_categorical(y_train_vote_fractions), \
  
                                               (x_test, np_utils.to_categorical(y_test)), \
                                               (x_test, np_utils.to_categorical(y_test)), \
                                                pretrained_weights=dec_weights, maxiter=1, \
                                                alpha=K.variable(1.0), beta=K.variable(0.0), gamma=K.variable(1.0),  \
                                                loss_weight_decay=False, update_interval=update_interval)

def learning_curve():
  plt.rcParams.update({'font.size': 12})
  x_train, y_train, x_train_dev, y_train_dev, x_valid, y_valid, x_test, y_test, order, split1, split2 = \
    load_data_set()
  
  y_train_first_human, y_test_first_human = get_labels_first_human()
  
  y_train_vote_fractions, y_train_dev_vote_fractions, y_test_vote_fractions = \
    get_labels_vote_fractions(order, split1, split2)

  all_ones_f1_benchmark, unsupervised_f1c_benchmark, first_human_f1_benchmark, vote_fraction_f1_benchmark, threepi_cnn_test_f1_benchmark, human_expert_f1_benchmark = \
    bench_marks(x_train, y_train, x_test, y_test, y_test_first_human, y_train_vote_fractions, y_test_vote_fractions)

  print(all_ones_f1_benchmark)
  print(unsupervised_f1c_benchmark)
  print(first_human_f1_benchmark)
  print(vote_fraction_f1_benchmark)
  metrics_dict = pickle.load(open('/Users/dwright/dev/zoo/machine_augmented_classification/notebooks/metrics_dict_paper_4.pkl', 'rb'))

  best_ite = 8
  print(metrics_dict['valid_f1c'])
  fig = plt.figure()
  ax1 = fig.add_subplot(121)
  ax1.set_title('multitask learning step')
  ax1.plot(metrics_dict['iteration'][:best_ite+1], metrics_dict['train_f1c'][:best_ite+1], label='train')
  ax1.plot(metrics_dict['iteration'][:best_ite+1], metrics_dict['valid_f1c'][:best_ite+1], label='validation')
  ax1.set_xlim(0,best_ite)
  ax1.set_ylim(-0.01,1.01)
  ax1.set_xlabel('iteration')
  ax1.set_ylabel('F1 score')
  ax1.legend(loc='lower right')

  ax2 = fig.add_subplot(121, sharex=ax1, frameon=False)
  ax2.yaxis.tick_right()
  ax2.yaxis.set_label_position("right")
  #ax2.plot(metrics_dict['iteration'], best_mapping_benchmark*np.ones(np.array(metrics_dict['iteration']).shape), 'k--')
  ax2.plot(metrics_dict['iteration'], first_human_f1_benchmark*np.ones(np.array(metrics_dict['iteration']).shape), 'k--')
  ax2.plot(metrics_dict['iteration'], vote_fraction_f1_benchmark*np.ones(np.array(metrics_dict['iteration']).shape), 'k--')
  ax2.plot(metrics_dict['iteration'], all_ones_f1_benchmark*np.ones(np.array(metrics_dict['iteration']).shape), 'k--')
  ax2.plot(metrics_dict['iteration'], unsupervised_f1c_benchmark*np.ones(np.array(metrics_dict['iteration']).shape), 'k--')
  ax2.plot(metrics_dict['iteration'], threepi_cnn_test_f1_benchmark*np.ones(np.array(metrics_dict['iteration']).shape), 'k--')
  ax2.set_xlim(0,best_ite)
  ax2.set_ylim(-0.01,1.01)
  ax2.yaxis.set_ticks_position('none')
  #ax2.set_yticks([all_ones_benchmark, best_mapping_benchmark, human_f1_benchmark, human_vote_fraction_benchmark, threepi_cnn_test_f1_benchmark])
  #ax2.set_yticks([all_ones_f1_benchmark, first_human_f1_benchmark, vote_fraction_f1_benchmark, threepi_cnn_test_f1_benchmark])
  #ax2.set_yticklabels(['all ones', 'best mapping', 'single volunteer', 'vote fractions', 'CNN'])
  ax2.set_yticklabels(['', '', '', '', ''])
  #ax2.set_yticklabels(['all ones', 'single volunteer', 'vote fractions', 'CNN'])

  ax3 = fig.add_subplot(122)
  ax3.set_title('reclustering step')
  ax3.plot(metrics_dict['iteration'][best_ite+1:], metrics_dict['train_f1c'][best_ite+1:], label='train + train-validation')
  ax3.plot(metrics_dict['iteration'][best_ite+1:], metrics_dict['valid_f1c'][best_ite+1:], label='validation')
  ax3.set_xlim(10,metrics_dict['iteration'][-1]+10)
  ax3.set_ylim(-0.01,1.01)
  ax3.yaxis.set_ticks_position('none')
  ax3.yaxis.set_ticklabels([])
  #ax3.set_xscale('log')
  ax3.set_xlabel('iteration')
  ax3.legend(loc='lower right')
  #ax3.set_ylabel('clustering f1 score')

  ax4 = fig.add_subplot(122, sharex=ax3, frameon=False)
  ax4.yaxis.tick_right()
  ax4.yaxis.set_label_position("right")
  #ax2.plot(metrics_dict['iteration'], best_mapping_benchmark*np.ones(np.array(metrics_dict['iteration']).shape), 'k--')
  ax4.plot(metrics_dict['iteration'][best_ite+1:], first_human_f1_benchmark*np.ones(np.array(metrics_dict['iteration'][best_ite+1:]).shape), 'k--')
  ax4.plot(metrics_dict['iteration'][best_ite+1:], vote_fraction_f1_benchmark*np.ones(np.array(metrics_dict['iteration'][best_ite+1:]).shape), 'k--')
  ax4.plot(metrics_dict['iteration'][best_ite+1:], all_ones_f1_benchmark*np.ones(np.array(metrics_dict['iteration'][best_ite+1:]).shape), 'k--')
  ax4.plot(metrics_dict['iteration'][best_ite+1:], unsupervised_f1c_benchmark*np.ones(np.array(metrics_dict['iteration'][best_ite+1:]).shape), 'k--')
  ax4.plot(metrics_dict['iteration'][best_ite+1:], threepi_cnn_test_f1_benchmark*np.ones(np.array(metrics_dict['iteration'][best_ite+1:]).shape), 'k--')
  ax4.set_xlim(10,metrics_dict['iteration'][-1]+10)
  ax4.set_ylim(-0.01,1.01)
  ax4.set_yticks([all_ones_f1_benchmark, unsupervised_f1c_benchmark, first_human_f1_benchmark, vote_fraction_f1_benchmark, threepi_cnn_test_f1_benchmark])
  ax4.set_yticklabels(['all ones', 'unsupervised', 'single volunteer', 'vote fractions', 'CNN'])
  plt.show()

def kmeans_comparison():
  x_train, y_train, x_train_dev, y_train_dev, x_valid, y_valid, x_test, y_test, order, split1, split2 = \
    load_data_set()
  
  y_train_first_human, y_test_first_human = get_labels_first_human()
  
  y_train_vote_fractions, y_train_dev_vote_fractions, y_test_vote_fractions = \
    get_labels_vote_fractions(order, split1, split2)
  """
  # load the pretrained DEC model for Supernova Hunters
  ae_weights  = '/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras/results/snhunters/train_only/ae_weights.h5'
  dec = MultitaskDEC(dims=[x_train.shape[-1], 500, 500, 2000, 10], \
                   n_clusters=n_clusters, batch_size=batch_size)
  dec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
                       ae_weights=ae_weights,
                       x=x_test)
  """
  x = np.concatenate((x_train, x_train_dev))
  y = np.concatenate((y_train_vote_fractions, y_train_dev_vote_fractions))
  kmeans = KMeans(n_clusters=n_clusters, n_init=20)
  #y_pred = kmeans.fit_predict(dec.encoder.predict(x))
  y_pred = kmeans.fit_predict(x)

  c_map, _, _ = \
    get_cluster_to_label_mapping_safe(y, y_pred, n_classes, n_clusters, toprint=True)
  #test_cluster_pred = kmeans.predict(dec.encoder.predict(x_test))
  test_cluster_pred = kmeans.predict(x_test)
  print(np.round(calc_f1_score(y_test, test_cluster_pred, c_map), 5))
  print(np.round(homogeneity_score(y_test, test_cluster_pred), 5))
  print(np.round(normalized_mutual_info_score(y_test, test_cluster_pred), 5))

import itertools
import operator

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

def count_clicks_simulation(y, y_pred, n_classes, n_clusters, sample_size = 25):
  import random
  click_counter = 0
  not_seen = [x for x in range(len(y))]
  while not_seen != []:
    for i in range(n_clusters):
      assigned = np.where(y_pred == i)[0].tolist()
      #print(assigned)
      assigned = [x for x in assigned if x in not_seen]
      if assigned == []:
        continue
      try:
        sample = random.sample(assigned, sample_size)
      except ValueError:
        sample = assigned
      not_seen = [x for x in not_seen if x not in sample]
      mc = most_common(y[sample])
      #print(np.where(y[sample] == mc)[0])
      click_counter += len(sample) - len(np.where(y[sample] == mc)[0])
  print(click_counter)
  return click_counter

def efficency_study_dec(x, y):
  # initial label gathering
  # load the pretrained DEC model for Supernova Hunters
  ae_weights  = '/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras/results/snhunters/train_only/ae_weights.h5'
  dec_weights = '/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras/results/snhunters/train_only/10/DEC_model_final.h5'
  dec = MultitaskDEC(dims=[x.shape[-1], 500, 500, 2000, 10], \
                   n_clusters=n_clusters, batch_size=batch_size)
  dec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
                       ae_weights=ae_weights,
                       x=x)
  dec.model.load_weights(dec_weights)
  n_trials = 5
  total = 0
  for i in range(n_trials):
    total += count_clicks_simulation(y, dec.predict_clusters(x), n_classes, n_clusters, sample_size = 25)
  print(total / float(n_trials))

def efficency_study_kmeans(x, y, x_test, y_test):
  from sklearn.cluster import KMeans
  ae_weights  = '/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras/results/snhunters/train_only/ae_weights.h5'
  dec = MultitaskDEC(dims=[x.shape[-1], 500, 500, 2000, 10], \
                     n_clusters=n_clusters, batch_size=batch_size)
  dec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
                       ae_weights=ae_weights,
                       x=x)

  kmeans = KMeans(n_clusters=n_clusters, n_init=20)
  kmeans.fit(dec.encoder.predict(x))

  c_map, _, _ = \
    get_cluster_to_label_mapping_safe(y, kmeans.predict(dec.encoder.predict(x)), n_classes, n_clusters, toprint=True)
  test_cluster_pred = kmeans.predict(dec.encoder.predict(x_test))
  print(np.round(calc_f1_score(y_test, test_cluster_pred, c_map), 5))
  print(np.round(homogeneity_score(y_test, test_cluster_pred), 5))
  print(np.round(normalized_mutual_info_score(y_test, test_cluster_pred), 5))
  print(np.round(get_cluster_to_label_mapping_safe(y_test, test_cluster_pred, n_classes, n_clusters), 5))
  n_trials = 5
  total = 0
  for i in range(n_trials):
    total += count_clicks_simulation(y, kmeans.predict(dec.encoder.predict(x)), n_classes, n_clusters, sample_size = 25)
  print(total / float(n_trials))

def efficency_study_proposed(x, y):
  # initial label gathering
  # load the pretrained DEC model for Supernova Hunters
  ae_weights  = '/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras/results/snhunters/train_only/ae_weights.h5'
  dec_weights = '/Users/dwright/dev/zoo/machine_augmented_classification/notebooks/paper/reclustering_step/results/dec/DEC_model_final.h5'
  redec = ReDEC(dims=[x.shape[-1], 500, 500, 2000, 10], \
                      n_clusters=n_clusters, batch_size=batch_size)
  redec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
                         ae_weights=ae_weights,
                         x=x)
  redec.model.load_weights(dec_weights)
  n_trials = 5
  total = 0
  for i in range(n_trials):
    total += count_clicks_simulation(y, redec.predict_clusters(x), n_classes, n_clusters, sample_size = 25)
  print(total / float(n_trials))

def efficency_study_proposed(x, y, x_test, y_test):
  # initial label gathering
  # load the pretrained DEC model for Supernova Hunters
  ae_weights  = '/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras/results/snhunters/train_only/ae_weights.h5'
  dec_weights = '/Users/dwright/dev/zoo/machine_augmented_classification/notebooks/paper/reclustering_step/results/dec/DEC_model_final.h5'
  redec = ReDEC(dims=[x.shape[-1], 500, 500, 2000, 10], \
                      n_clusters=n_clusters, batch_size=batch_size)
  redec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
                         ae_weights=ae_weights,
                         x=x)
  redec.model.load_weights(dec_weights)
  n_trials = 5
  total = 0
  """
  for i in range(n_trials):
    total += count_clicks_simulation(y, redec.predict_clusters(x), n_classes, n_clusters, sample_size = 25)
  print(total / float(n_trials))
  """
  for i in range(n_trials):
    total += count_clicks_simulation(y_test, redec.predict_clusters(x_test), n_classes, n_clusters, sample_size = 25)
  print(total / float(n_trials))

def efficency_study_kmeans_recluster(x_train, y_train, x_train_dev, y_train_dev, x_valid, y_valid, x_test, y_test):
  # initial label gathering
  # load the pretrained DEC model for Supernova Hunters
  
  ae_weights  = '/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras/results/snhunters/train_only/ae_weights.h5'
  dec = MultitaskDEC(dims=[x_train.shape[-1], 500, 500, 2000, 10], \
                   n_clusters=n_clusters, batch_size=batch_size)
  dec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
                       ae_weights=ae_weights,
                       x=x_train)
  
  ## !!!!! Make sure to turn off loading of dec_weights in
  dec.clustering(x_train, np_utils.to_categorical(y_train), \
                (x_train_dev, np_utils.to_categorical(y_train_dev)), \
                (x_valid, np_utils.to_categorical(y_valid)), \
                (x_test, np_utils.to_categorical(y_test)), \
                pretrained_weights=None, maxiter=5, \
                alpha=K.variable(1.0), beta=K.variable(0.0), gamma=K.variable(1.0),  \
                loss_weight_decay=False, update_interval=update_interval, \
                save_dir='./results/kmeans/dec/')
                
  dec.model.load_weights('/Users/dwright/dev/zoo/machine_augmented_classification/paper/results/kmeans/dec/best_train_dev_loss_kmeans.hf', by_name=True)

  x = np.concatenate((x_train, x_train_dev))
  y = np.concatenate((y_train, y_train_dev))
  print(x.shape)
  print(y.shape)
  kmeans = KMeans(n_clusters=n_clusters, n_init=20)
  kmeans.fit(dec.encoder.predict(x))
  """
  c_map, _, _, _ = \
    get_cluster_to_label_mapping_safe(y, dec.encoder.predict(x), n_classes, n_clusters, toprint=False)
  test_cluster_pred = kmeans.predict(dec.encoder.predict(x_test))
  #test_cluster_pred = kmeans.predict(x_test)
  print(np.round(calc_f1_score(y_test, test_cluster_pred, c_map), 5))
  print(np.round(homogeneity_score(y_test, test_cluster_pred), 5))
  print(np.round(normalized_mutual_info_score(y_test, test_cluster_pred), 5))
  print(np.round(get_cluster_to_label_mapping_safe(y_test, test_cluster_pred, n_classes, n_clusters), 5))
  """
  n_trials = 5
  total = 0
  for i in range(n_trials):
    total += count_clicks_simulation(y, kmeans.predict(dec.encoder.predict(x)), n_classes, n_clusters, sample_size = 25)
  print(total / float(n_trials))

from sklearn.decomposition import PCA
def pca_plot(base_network, x, cluster_centres=None, y=None, labels=[], output_file=None,\
             lcolours=[], ulcolour='#747777', ccolour='#4D6CFA', legend=False):
    
  def onpick(event):
    print('picked')
    print(event.ind)
    print(y[event.ind[0]])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.reshape(x[event.ind[0]], (20,20)), cmap='gray_r')
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

def more_data_test(x_train, y_train, x_train_dev, y_train_dev, x_valid, y_valid, x_test, y_test, y_train_vote_fractions, y_train_dev_vote_fractions, y_test_vote_fractions):

  vote_fraction_dict = pickle.load(open('/Users/dwright/dev/zoo/data/supernova-hunters-vote-fraction-dict.pkl', 'rb'))
  data = sio.loadmat('/Users/dwright/dev/zoo/machine_augmented_classification/data/snhunters/3pi_20x20_supernova_hunters_batch_10_signPreserveNorm_detect_misaligned.mat')
  ae_weights  = '/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras/results/snhunters/train_only/ae_weights.h5'
  x = np.nan_to_num(np.reshape(data['X'], (data['X'].shape[0], 400), order='F'))
  y = np.squeeze(data['y'])
  u, indices = np.unique(x, return_index=True, axis=0)
  x = x[indices]
  y = y[indices]
  files = []
  for f in data['files'][indices]:
    files.append(f.strip().replace('.fits',''))
  vote_fractions_x = []
  seen = []
  for diff in files:
    vote_fractions_x.append(vote_fraction_dict[diff])
    seen.append(diff)
  print(vote_fractions_x[:10])
  vote_fractions_x = np.array(vote_fractions_x)
  y_train_vote_x = vote_fractions_x > 0.5
  m = x.shape[0]
  order = np.random.permutation(m)
  x = x[order]
  y = y[order]
  y_train_vote_x = y_train_vote_x[order]

  x_train_dev_x = x[int(.667*m):]
  y_train_dev_vote_x = y_train_vote_x[int(.667*m):]

  x = np.concatenate((x[:int(.667*m)], x_train, x_train_dev))
  y_train_vote_x = np.concatenate((y_train_vote_x[:int(.667*m)], y_train_vote_fractions, y_train_dev_vote_fractions))
  dec = MultitaskDEC(dims=[x_train.shape[-1], 500, 500, 2000, 10], \
                   n_clusters=n_clusters, batch_size=batch_size)
  dec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
                       ae_weights=ae_weights,
                       x=x_train)
  dec.model.load_weights('/Users/dwright/dev/zoo/machine_augmented_classification/notebooks/paper/reclustering_step/results/dec/DEC_model_final.h5')

  pca_plot(dec.encoder, x, y=y_train_vote_x, \
         cluster_centres=get_cluster_centres(dec), labels=['bogus', 'real'], lcolours=['#D138BF','#7494EA'], \
         ulcolour='#A0A4B8', ccolour='#21D19F', legend=False)


  y_pred, metrics_dict, best_ite = dec.clustering(x, np_utils.to_categorical(y_train_vote_x), \
                                               (x_train_dev_x, np_utils.to_categorical(y_train_dev_vote_x)), \
                                               (x_valid, np_utils.to_categorical(y_valid)), \
                                                (x_test, np_utils.to_categorical(y_test)), \
                                                pretrained_weights='/Users/dwright/dev/zoo/machine_augmented_classification/notebooks/paper/reclustering_step/results/dec/DEC_model_final.h5', maxiter=maxiter, \
                                                alpha=K.variable(1.0), beta=K.variable(0.0), gamma=K.variable(1.0),  \
                                                loss_weight_decay=False, update_interval=update_interval, \
                                                save_dir='./paper/multitask_step/zoo_updates/results/dec/')

  redec = ReDEC(dims=[x.shape[-1], 500, 500, 2000, 10], \
                     n_clusters=n_clusters, batch_size=batch_size)
  redec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
                        ae_weights=ae_weights,
                       x=x)
  dec.model.load_weights('./paper/multitask_step/zoo_updates/results/dec/best_train_dev_loss_10000_zoo_updated.hf', by_name=True)
  dec.model.summary()
  for i in range(1,len(dec.model.layers[1].layers)):
    redec.model.layers[i].set_weights(dec.model.layers[1].layers[i].get_weights())
  redec.model.layers[-1].set_weights(dec.model.layers[2].get_weights())
  y_pred, metrics_dict = redec.clustering((np.concatenate((x, x_train_dev_x)), np.concatenate((y_train_vote_x, y_train_dev_vote_x))), \
                                        (x_valid, y_valid), (x_test, y_test), pretrained_weights=None, \
                                        maxiter=2e4, update_interval=140, metrics_dict=metrics_dict, last_ite=best_ite, \
                                        save_dir='./paper/reclustering_step/zoo/results/dec/')


def main():
  #learning_curve()
  #kmeans_comparison()
  
  x_train, y_train, x_train_dev, y_train_dev, x_valid, y_valid, x_test, y_test, order, split1, split2 = \
    load_data_set()
  #y_train_first_human, y_test_first_human = get_labels_first_human()
  
  y_train_vote_fractions, y_train_dev_vote_fractions, y_test_vote_fractions = \
    get_labels_vote_fractions(order, split1, split2)
  #get_initial_results(x_train,y_train_vote_fractions,x_test,y_test)
  #get_multitask_step_results(x_train, y_train_vote_fractions, x_test, y_test)

  #efficency_study_dec(np.concatenate((x_train, x_train_dev)), \
  #                    np.concatenate((y_train, y_train_dev)), \
  #                    x_test, y_test)
  #efficency_study_kmeans(np.concatenate((x_train, x_train_dev)), \
  #                       np.concatenate((y_train, y_train_dev)), \
  #                       x_test, y_test)
  #efficency_study_proposed(np.concatenate((x_train, x_train_dev)), \
  #                         np.concatenate((y_train, y_train_dev)), \
  #                         x_test, y_test)
  #efficency_study_kmeans_recluster(x_train, y_train, x_train_dev, y_train_dev, x_valid, y_valid, x_test, y_test)
  #clickable_analysis(x_train, y_train, x_train_dev, y_train_dev, x_valid, y_valid, x_test, y_test)
  more_data_test(x_train, y_train, x_train_dev, y_train_dev, x_valid, y_valid, x_test, y_test, y_train_vote_fractions, y_train_dev_vote_fractions, y_test_vote_fractions)
if __name__ == '__main__':
  main()
