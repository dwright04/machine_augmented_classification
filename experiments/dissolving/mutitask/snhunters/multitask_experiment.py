import sys
import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, roc_curve
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import homogeneity_score

from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Dropout
from keras.initializers import Initializer
from keras.optimizers import SGD, Adadelta
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras.engine.topology import Layer
from keras import backend as K
from keras import regularizers
#from keras.models import load_model

sys.path.insert(0,'/Users/dwright/dev/zoo/machine_augmented_classification/DEC-keras')
from DEC import DEC, ClusteringLayer, cluster_acc

sys.path.insert(0,'/Users/dwright/dev/zoo/machine_augmented_classification/experiments/dissolving')
from dissolving_utils import get_cluster_centres, get_cluster_to_label_mapping
from dissolving_utils import pca_plot
from dissolving_utils import MappingLayer, MapInitializer
from dissolving_utils import load_dec

lcolours = ['#D6FF79', '#B0FF92', '#A09BE7', '#5F00BA', '#56CBF9', \
            '#F3C969', '#ED254E', '#CAA8F5', '#D9F0FF', '#46351D']

# DEC constants from DEC paper
batch_size = 256
lr         = 0.01
momentum   = 0.9
tol        = 0.001
maxiter    = 100
update_interval = 140 #perhaps this should be 1 for multitask learning
#update_interval = 10 #perhaps this should be 1 for multitask learning
n_clusters = 10 # number of clusters to use
n_classes  = 2  # number of classes

class MyLossWeightCallback(Callback):
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    # customize your behavior
    def on_epoch_begin(self, epoch, logs={}):
        self.alpha = self.alpha
        self.beta = self.beta
        self.gamma = self.gamma

class MultitaskDEC(DEC):
        
 def clustering(self, x, y=None, train_dev_data=None, validation_data=None, tol=1e-3, update_interval=140, maxiter=2e4, save_dir='./results/dec', pretrained_weights=None, alpha=K.variable(1.0), beta=K.variable(0.0), gamma=K.variable(0.0),  loss_weight_decay=True, loss=None, loss_weights=None):
    print('Update interval', update_interval)
    save_interval = x.shape[0] / self.batch_size * 5  # 5 epochs
    print('Save interval', save_interval)

    try:
      self.load_weights(pretrained_weights)
    except AttributeError:
      # initialize cluster centers using k-means
      print('Initializing cluster centers with k-means.')
      kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
      y_pred = kmeans.fit_predict(self.encoder.predict(x))
      y_pred_last = y_pred
      self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    y_p = self.predict_clusters(x)

    cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
      get_cluster_to_label_mapping_safe(y[:,1], y_p, n_classes, n_clusters)
    
    print(np.argmax((1-np.array(majority_class_fractions))*np.array(n_assigned_list)))
    cluster_to_label_mapping[np.argmax((1-np.array(majority_class_fractions))*np.array(n_assigned_list))] = 1
    
    cluster_weights = self.model.get_layer(name='clustering').get_weights()
    #self.tmp_model = Model(inputs=self.model.input, outputs=self.model.layers[-2].output)
    a = Input(shape=(400,)) # input layer
    #self.model.summary()
    #self.model.layers[1].trainable = False
    #self.model.layers[2].trainable = False
    #self.model.layers[3].trainable = False
    self.model.layers[1].kernel_regularizer = regularizers.l2(0.5)
    self.model.layers[2].kernel_regularizer = regularizers.l2(0.5)
    self.model.layers[3].kernel_regularizer = regularizers.l2(0.5)
    self.model.layers[4].kernel_regularizer = regularizers.l2(0.5)
    #self.model.summary()
    #self.autoencoder.summary()
    #self.encoder.summary()
    #exit()
    #q_out = self.model(b)
    hidden = self.encoder(a)
    q_out = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
    #d = Dropout(0.2)(a)
    e_out = self.autoencoder(a)
    #pred = MappingLayer(cluster_to_label_mapping, output_dim=n_classes, \
    #  name='mapping', kernel_initializer=MapInitializer(cluster_to_label_mapping, n_classes))(q_out)
    #embed = self.encoder(a)
    #d_out = Dropout(0.4)(embed)
    #self.encoder.summary()
    #d_out = Dropout(0.7)(hidden)
    #pred = MappingLayer(cluster_to_label_mapping, output_dim=n_classes, \
    #  name='mapping', kernel_initializer=MapInitializer(cluster_to_label_mapping, n_classes))(d_out)
    #pred = Dense(2, activation='softmax')(hidden)
    pred = Dense(2, activation='softmax')(q_out)
    #pred = Dense(2, activation='softmax')(d_out)
    self.model = Model(inputs=a, outputs=[pred, q_out, e_out])
    #self.model = Model(inputs=a, outputs=[pred, q_out])
    self.model.get_layer(name='clustering').set_weights(cluster_weights)
    #optimizer = SGD(lr=1e-2)
    optimizer = 'adam'
    #self.model.compile(optimizer=optimizer, \
    #                   loss={'dense_1': 'categorical_crossentropy', 'clustering': 'kld'}, \
    #                   loss_weights={'dense_1': alpha, 'clustering': beta})
    if loss is None:
      self.model.compile(optimizer=optimizer, \
                         loss={'dense_1': 'categorical_crossentropy', 'clustering': 'kld', 'model_1': 'mse'},
                         loss_weights={'dense_1': alpha, 'clustering': beta, 'model_1': gamma})
    else:
      self.model.compile(optimizer=optimizer, \
                         loss=loss,
                         loss_weights=loss_weights)
    #self.model.compile(optimizer=optimizer, \
    #                   loss={'mapping': 'categorical_crossentropy', 'model_3': 'kld'}, \
    #                   loss_weights={'mapping': alpha, 'model_3': beta})
    import csv, os
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    #logfile = open(save_dir + '/dec_log.csv', 'w')
    #logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L'])
    #logwriter.writeheader()
    #tmp = self.encoder.predict(x)
    loss = [0, 0, 0]
    index = 0
    q = self.model.predict(x, verbose=0)[1]
    y_pred_last = q.argmax(1)
    metrics_dict = {'iteration':[],
                    'train_fom':[],
                    'train_f1':[],
                    'train_f1c':[],
                    'train_h':[],
                    'train_nmi':[],
                    'valid_fom':[],
                    'valid_f1':[],
                    'valid_f1c':[],
                    'valid_h':[],
                    'valid_nmi':[]
                    }
    best_train_dev_loss = [np.inf, np.inf, np.inf]
    for ite in range(int(maxiter)):
      if ite % update_interval == 0:
        q = self.model.predict(x, verbose=0)[1]
        valid_p = self.target_distribution(self.model.predict(validation_data[0], verbose=0)[1])
        p = self.target_distribution(q)  # update the auxiliary target distribution p
        #print(p)
        #print(np.sum(self.encoder.predict(x) - tmp))
        #print(np.sum(self.encoder.predict(x) - self.tmp_model.predict(x)))
        
        # evaluate the clustering performance
        y_pred = q.argmax(1)
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = y_pred
        y_pred = self.model.predict(x)[0]
        if y is not None:
          #acc = np.round(cluster_acc(y, y_pred), 5)
          #ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
          loss = np.round(loss, 5)
          #val_loss = np.round(self.model.test_on_batch(validation_data[0], [validation_data[1], valid_p]), 5)
          c_map, _, _ = \
            get_cluster_to_label_mapping_safe(y[:,1], q.argmax(1), n_classes, n_clusters, toprint=False)
          f, _, _, _ = one_percent_fpr(y[:,1], y_pred[:,1], 0.01)
          f = np.round(f, 5)
          #print(y[:20,1], np.argmax(y_pred, axis=1)[:20])
          train_cluster_pred = self.model.predict(x, verbose=0)[1].argmax(1)
          f1 = np.round(f1_score(y[:,1], np.argmax(y_pred, axis=1)), 5)
          f1c = np.round(calc_f1_score(y[:,1], train_cluster_pred, c_map), 5)
          h = np.round(homogeneity_score(y[:,1], train_cluster_pred), 5)
          nmi = np.round(metrics.normalized_mutual_info_score(y[:,1], train_cluster_pred), 5)
          val_loss = np.round(self.model.test_on_batch(validation_data[0], [validation_data[1], valid_p, validation_data[0]]), 5)
          y_pred_valid = self.model.predict(validation_data[0])[0]
          valid_cluster_pred = self.model.predict(validation_data[0], verbose=0)[1].argmax(1)
          f_valid, _, _, _ = one_percent_fpr(validation_data[1][:,1], y_pred_valid[:,1], 0.01)
          f_valid = np.round(f_valid, 5)
          f1_valid = np.round(f1_score(validation_data[1][:,1], np.argmax(y_pred_valid, axis=1)), 5)
          f1c_valid = np.round(calc_f1_score(validation_data[1][:,1], valid_cluster_pred, c_map), 5)
          h_valid = np.round(homogeneity_score(validation_data[1][:,1], valid_cluster_pred), 5)
          nmi_valid = np.round(metrics.normalized_mutual_info_score(validation_data[1][:,1], valid_cluster_pred), 5)
          #logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss)
          #logwriter.writerow(logdict)
          #print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)
          #_, _, _ = \
          #  get_cluster_to_label_mapping_safe(validation_data[1][:,1], self.model.predict(validation_data[0], verbose=0)[1].argmax(1), n_classes, n_clusters)
          print('Iter', ite, ' :1% fpr', f, ', F1=', f1, ', F1c=', f1c, 'h=', h, 'nmi=', nmi, '; loss=', loss, ';\n v 1% fpr=,', f_valid, ', vF1=', f1_valid, ', vF1c=', f1c_valid,'vh=', h_valid, 'vnmi=', nmi_valid, '; vloss=,', val_loss)
          metrics_dict['iteration'].append(ite)
          metrics_dict['train_fom'].append(f)
          metrics_dict['train_f1'].append(f1)
          metrics_dict['train_f1c'].append(f1c)
          metrics_dict['train_h'].append(h)
          metrics_dict['train_nmi'].append(nmi)
          metrics_dict['valid_fom'].append(f_valid)
          metrics_dict['valid_f1'].append(f1_valid)
          metrics_dict['valid_f1c'].append(f1c_valid)
          metrics_dict['valid_h'].append(h_valid)
          metrics_dict['valid_nmi'].append(nmi_valid)
          
          train_dev_p = self.target_distribution(self.model.predict(train_dev_data[0], verbose=0)[1])
          train_dev_loss = np.round(self.model.test_on_batch(train_dev_data[0], [train_dev_data[1], train_dev_p, train_dev_data[0]]), 5)
          if train_dev_loss[1] < best_train_dev_loss[1] and train_dev_loss[-1] < best_train_dev_loss[-1]: # only interested in classification improvements
            print('saving model: ', best_train_dev_loss, ' -> ', train_dev_loss)
            self.model.save_weights('best_train_dev_loss.hf')
            best_train_dev_loss = train_dev_loss
            best_ite = ite
      
        # check stop criterion
        
        if ite > 0 and delta_label < tol:
          print('delta_label ', delta_label, '< tol ', tol)
          print('Reached tolerance threshold. Stopping training.')
          logfile.close()
          break
        
        # train on batch
        """
        if (index + 1) * self.batch_size > x.shape[0]:
          loss = self.model.train_on_batch(x=x[index * self.batch_size::],
                                           y=[y[index * self.batch_size::], \
                                              p[index * self.batch_size::]])
          index = 0
        else:
          loss = self.model.train_on_batch(x=x[index * self.batch_size:(index + 1) * self.batch_size],
                                           y=[y[index * self.batch_size:(index + 1) * self.batch_size], \
                                              p[index * self.batch_size:(index + 1) * self.batch_size]])
          index += 1
        """
      
      if loss_weight_decay:
        """
        if ite < 50:
          alpha = K.variable(1.0)
          beta  = K.variable(0.0)
          gamma = K.variable(1.0)
        elif ite >= 50:
          #alpha = K.variable(1.0)
          alpha = K.variable(0.0)
          #beta  = K.variable(0.0)
          beta  = K.variable(1.0)
          gamma  = K.variable(1.0)
          update_interval = 140
          self.model.optimizer = SGD(lr=0.01, momentum=0.9)
        """
        """
        elif ite >= 200 and ite < 300:
          #alpha = K.variable(1.0*(1 - ((ite - 200)/100.)))
          alpha = K.variable(1.0)
          beta  = K.variable(1.0)
          gamma = K.variable(1.0)
        print(K.eval(alpha), K.eval(beta))
        """
        #alpha = K.variable(1.0*(1 - ((ite - 200)/100.)))
        """
        if ite < 40:
          alpha = K.variable((1 - ite/maxiter))
          beta  = K.variable(1-alpha)
          gamma = K.variable(1.0)
        print(K.eval(alpha), K.eval(beta), K.eval(gamma))
        if ite == 40:
          print('Initializing cluster centers with k-means.')
          kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
          y_pred = kmeans.fit_predict(self.encoder.predict(x))
          self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        if ite >= 40:
          alpha = K.variable(0.0)
          beta  = K.variable(1.0)
          gamma = K.variable(1.0)
          update_interval=140
          self.model.optimizer = SGD(lr=0.01, momentum=0.9)
        """
        
        alpha = K.variable((1 - ite/maxiter))
        beta  = K.variable(1-alpha)
        gamma = K.variable(1.0)
        print(K.eval(alpha), K.eval(beta), K.eval(gamma))
        history = self.model.fit(x=x, y=[y,p,x], \
                  validation_data=(validation_data[0], [validation_data[1], valid_p, validation_data[0]]), \
                  callbacks=[MyLossWeightCallback(alpha, beta, gamma)], verbose=0)
      else:
        print(K.eval(alpha), K.eval(beta), K.eval(gamma))
        history = self.model.fit(x=x, y=[y,p,x], \
                  validation_data=(validation_data[0], [validation_data[1], valid_p, validation_data[0]]), \
                  verbose=0)
      #history = self.model.fit(x=x, y=[y,p], callbacks=[MyLossWeightCallback(alpha, beta)], verbose=0)
      #print(history.history)
      loss = [history.history[k][0] for k in history.history.keys() if 'val' not in k]
        # save intermediate model
      if ite % save_interval == 0:
        # save IDEC model checkpoints
        print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
        self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')

      ite += 1

    # save the trained model
    #logfile.close()
    print('saving model to:', save_dir + '/DEC_model_final.h5')
    self.model.save_weights(save_dir + '/DEC_model_final.h5')

    y_p = self.model.predict(x, verbose=0)[1].argmax(1)
    cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
      get_cluster_to_label_mapping_safe(y[:,1], y_p, n_classes, n_clusters)
    return y_pred, metrics_dict, best_ite

def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

def custom_loss(y_true, y_pred):
   #print(labelled_limit)
   #print(K.shape(y_true), K.shape(y_pred))
   kld_loss = kullback_leibler_divergence(y_true, y_pred)
   #print(K.shape(kld_loss))
   #print(K.shape(y_true[:labelled_limit]), K.shape(y_pred[:labelled_limit]))
   cc_loss = K.categorical_crossentropy(y_true[:labelled_limit], y_pred[:labelled_limit])
   
   print(K.shape(cc_loss), K.shape(kld_loss))
   
   return  cc_loss + 0.01 * kld_loss

def calc_fixed_fpr(y_true, y_preds, cluster_to_label_mapping, fom=0.1):
  #print(y_preds[:,np.where(np.array(cluster_to_label_mapping)==1)[0]].shape)
  #print(np.sum(y_preds[:,np.where(np.array(cluster_to_label_mapping)==1)[0]], axis=1).shape)
  #print(y_true.shape)
  return one_percent_fpr(y_true, np.sum(y_preds[:,np.where(np.array(cluster_to_label_mapping)==1)[0]], axis=1), fom)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), \
                            K.epsilon()))

class MultitaskLabelsByClusterDEC(DEC):

 def cluster_assignment(self, x, y, y_p, cluster_centres, cluster_to_label_mapping):
    x_encoded = self.encoder.predict(x)
    x_encoded_tiled = np.tile(x_encoded[:,:,np.newaxis], (1,1,self.n_clusters))
    cluster_centres_tiled = np.tile(cluster_centres[np.newaxis,:,:], (x.shape[0],1,1))
    euclidean_distances = np.squeeze(K.eval(euclidean_distance((x_encoded_tiled, cluster_centres_tiled))))
    cluster_preds = y_p
    y_c = []
    for i in range(x.shape[0]):
      l = np.argmax(y[i])
      c = cluster_preds[i]
      cl = cluster_to_label_mapping[c]
      if l == cl:
        y_c.append(c)
      else:
        ed = euclidean_distances[i][[np.where(cluster_to_label_mapping == l)]]
        ac = int(np.array(cluster_to_label_mapping)[np.where(cluster_to_label_mapping == l)][np.argmin(ed)])
        y_c.append(ac)
 
    # one hot encode these cluster assignements
    y_c = np_utils.to_categorical(y_c, self.n_clusters)
    return y_c
    
 def clustering(self, x, y, validation_data, tol=1e-3, update_interval=140, maxiter=2e4, save_dir='./results/dec', pretrained_weights=None, alpha=K.variable(0.9), beta=K.variable(0.1)):
 
    print('Update interval', update_interval)
    save_interval = x.shape[0] / self.batch_size * 5  # 5 epochs
    print('Save interval', save_interval)

    try:
      self.load_weights(pretrained_weights)
    except AttributeError:
      # initialize cluster centers using k-means
      print('Initializing cluster centers with k-means.')
      kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
      y_pred = kmeans.fit_predict(self.encoder.predict(x))
      y_pred_last = y_pred
      self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    # using only the labelled data set calculate the cluster label assignments
    y_pred = self.predict_clusters(x)
    self.n_classes = len(np.unique(y))
    cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
      get_cluster_to_label_mapping_safe(y[:,1], y_pred, self.n_classes, self.n_clusters)
    
    # ensure that the cluster with the most real detections is assigned the real class
    print(np.argmax((1-np.array(majority_class_fractions))*np.array(n_assigned_list)))
    cluster_to_label_mapping[np.argmax((1-np.array(majority_class_fractions))*np.array(n_assigned_list))] = 1
    cluster_centres = np.squeeze(np.array(self.model.get_layer(name='clustering').get_weights()))
    
    # build training set based on euclidean distances
    """
    x_encoded = self.encoder.predict(x_l)
    x_encoded_tiled = np.tile(x_encoded[:,:,np.newaxis], (1,1,self.n_clusters))
    cluster_centres_tiled = np.tile(cluster_centres[np.newaxis,:,:], (x_l.shape[0],1,1))
    euclidean_distances = np.squeeze(K.eval(euclidean_distance((x_encoded_tiled, cluster_centres_tiled))))
    cluster_preds = y_p
    y_c = []
    for i in range(x_l.shape[0]):
      l = np.argmax(y_l[i])
      c = cluster_preds[i]
      cl = cluster_to_label_mapping[c]
      if l == cl:
        y_c.append(c)
      else:
        ed = euclidean_distances[i][[np.where(cluster_to_label_mapping == l)]]
        ac = int(np.array(cluster_to_label_mapping)[np.where(cluster_to_label_mapping == l)][np.argmin(ed)])
        y_c.append(ac)
 
    # one hot encode these cluster assignements
    y_c = np_utils.to_categorical(y_c, self.n_clusters)
    """
    """
    # build the model
    a = Input(shape=(400,)) # input layer
    q_out = self.model(a)
    embed = self.encoder(a)
    pred = Dense(self.n_clusters, activation='softmax')(embed)
    self.model = Model(inputs=a, outputs=[q_out, pred])
    optimizer = SGD(lr=1e-1)
    #optimizer = 'adam'
    self.model.compile(optimizer=optimizer, \
                       loss={'dense_1': 'categorical_crossentropy', 'model_3': 'kld'}, \
                       loss_weights={'dense_1': alpha, 'model_3': beta})
    """
    a = Input(shape=(400,)) # input layer
    self.model.layers[1].kernel_regularizer = regularizers.l2(0.5)
    self.model.layers[2].kernel_regularizer = regularizers.l2(0.5)
    self.model.layers[3].kernel_regularizer = regularizers.l2(0.5)
    self.model.layers[4].kernel_regularizer = regularizers.l2(0.5)
    hidden = self.encoder(a)
    q_out = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
    e_out = self.autoencoder(a)
    pred = Dense(self.n_clusters, activation='softmax')(hidden)
    self.model = Model(inputs=a, outputs=[pred, q_out, e_out])
    self.model.get_layer(name='clustering').set_weights(cluster_weights)
    optimizer = SGD(lr=1e-1)
    self.model.compile(optimizer=optimizer, \
                       loss={'dense_1': 'categorical_crossentropy', 'clustering': 'kld', 'model_1': 'mse'}, \
                       loss_weights={'dense_1': alpha, 'clustering': beta, 'model_1': 1})
    
    
    loss = [0, 0, 0]
    index = 0
    q = self.model.predict(x, verbose=0)[1]
    y_pred_last = q.argmax(1)
    best_val_loss = [np.inf, np.inf, np.inf]
    for ite in range(int(maxiter)):
      if ite % update_interval == 0:
        q = self.model.predict(x, verbose=0)[1]
        p = self.target_distribution(q)  # update the auxiliary target distribution p
        
        # evaluate the clustering performance
        y_pred = q.argmax(1)
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = y_pred
        y_pred = self.model.predict(x)[0]
        #print(self.model.get_layer(name='model_3').layers[-1].get_weights()[0])
        y_c = self.cluster_assignment(x, y, p.argmax(1), self.model.get_layer(name='model_3').layers[-1].get_weights()[0], np.array(cluster_to_label_mapping))
        if y is not None:
          loss = np.round(loss, 5)
          valid_p = self.target_distribution(self.model.predict(validation_data[0], verbose=0)[1])
          y_c_valid = self.cluster_assignment(validation_data[0], validation_data[1], valid_p.argmax(1), self.model.get_layer(name='model_3').layers[-1].get_weights()[0], np.array(cluster_to_label_mapping))
          val_loss = np.round(self.model.test_on_batch(validation_data[0], [y_c_valid, valid_p]), 5)
          #val_loss = np.round(self.model.test_on_batch(validation_data[0], [validation_data[1], valid_p, validation_data[0]]), 5)
          #f, _, _, _ = one_percent_fpr(y[:,1], y_pred[:,1], 0.01)
          #f = np.round(f, 5)
          #print(y[:20,1], np.argmax(y_pred, axis=1)[:20])
          #f1 = np.round(f1_score(y[:,1], np.argmax(y_pred, axis=1)), 5)
          #f1 = np.round(calc_f1_score(y, q.argmax(1), cluster_to_label_mapping), 5)
          h = np.round(homogeneity_score(y[:,1], self.model.predict(x, verbose=0)[1].argmax(1)), 5)
          #y_pred_valid = self.model.predict(validation_data[0])[0]
          #f_valid, _, _, _ = one_percent_fpr(validation_data[1][:,1], y_pred_valid[:,1], 0.01)
          #f_valid = np.round(f_valid, 5)
          #f1_valid = np.round(f1_score(validation_data[1][:,1], np.argmax(y_pred_valid, axis=1)), 5)
          #f1_valid = np.round(calc_f1_score(validation_data[1], self.model.predict(validation_data[0], verbose=0)[1], cluster_to_label_mapping), 5)
          h_valid = np.round(homogeneity_score(validation_data[1][:,1], self.model.predict(validation_data[0], verbose=0)[1].argmax(1)), 5)
          #logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss)
          #logwriter.writerow(logdict)
          #print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)
          _, _, _ = \
            get_cluster_to_label_mapping_safe(y[:,1], q.argmax(1), n_classes, n_clusters)
          _, _, _ = \
            get_cluster_to_label_mapping_safe(validation_data[1][:,1], self.model.predict(validation_data[0], verbose=0)[1].argmax(1), n_classes, n_clusters)
          #print('Iter', ite, ' :1% fpr', f, ', F1=', f1, 'h=', h, '; loss=', loss, \
          #      ';\n\t\t valid 1% fpr=,', f_valid, ', valid F1=', f1_valid, 'h_valid=', h_valid, '; valid_loss=,', val_loss, )
          print('Iter', ite, 'h=', h, '; loss=', loss, 'h_valid=', h_valid, '; valid_loss=,', val_loss)
          if val_loss[1] < best_val_loss[1]: # only interested in classification improvements
            print('saving model: ', best_val_loss, ' -> ', val_loss)
            self.model.save_weights('best_val_loss.hf')
            best_val_loss = val_loss
      
        #alpha = K.variable(0.9*(1 - (ite/maxiter)))
        #beta  = K.variable(1 - alpha)
        print(K.eval(alpha), K.eval(beta))
        history = self.model.fit(x=x, y=[y_c,p], callbacks=[MyLossWeightCallback(alpha, beta)], verbose=0)
        #history = self.model.fit(x=x, y=[y,p,x], callbacks=[MyLossWeightCallback(alpha, beta)], verbose=0)
        print(history.history)
        loss = [history.history[k][0] for k in history.history.keys()]
        # save intermediate model
        if ite % save_interval == 0:
        # save IDEC model checkpoints
          print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
          self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')

        ite += 1

    # save the trained model
    #logfile.close()
    print('saving model to:', save_dir + '/DEC_model_final.h5')
    self.model.save_weights(save_dir + '/DEC_model_final.h5')

    return y_pred

def one_percent_fpr(y, pred, fom):
    fpr, tpr, thresholds = roc_curve(y, pred)
    FoM = 1-tpr[np.where(fpr<=fom)[0][-1]] # MDR at 1% FPR
    threshold = thresholds[np.where(fpr<=fom)[0][-1]]
    return FoM, threshold, fpr, tpr

def calc_f1_score(y_true, predicted_clusters, cluster_to_label_mapping):
  y_pred = []
  for i in range(len(y_true)):
    y_pred.append(cluster_to_label_mapping[predicted_clusters[i]])
  return f1_score(y_true, np.array(y_pred))

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
  return cluster_to_label_mapping, n_assigned_list, majority_class_fractions

def load_data(sample_frac):
  """
    !!!!! NEED TO CHANGE THIS TO ENSURE NO OVERLAP BETWEEN OBJECTS FROM SAME NIGHT IN
    VALIDATION AND TRAINING SETS !!!!!
  """
  # load snhunters data set
  data = sio.loadmat('../../../../data/3pi_20x20_skew2_signPreserveNorm.mat')
  #x = np.concatenate((data['X'], data['testX']))
  #y = np.squeeze(np.concatenate((data['y'], data['testy'])))
  x = data['X']
  y = data['y']
  # split the data into training, validation and test sets similar to MNIST
  m = data['X'].shape[0]
  m = m - int(.25*m)
  split = int(sample_frac*m)
  print(m,split)
  x_train = data['X'][:split]
  y_train = np.squeeze(data['y'])[:split]
  x_valid = data['X'][split:]
  y_valid = np.squeeze(data['y'])[split:]
  x_test  = data['testX']
  y_test  = np.squeeze(data['testy'])

  return x, y, m, split, x_train, y_train, x_valid, y_valid, x_test, y_test

def files_to_diff_ids(files):
  diff_ids = []
  for f in files:
    diff_ids.append(f.strip().replace('.fits',''))
  return diff_ids

def get_vote_fraction_labels(diff_ids):
  labels = []
  vote_fraction_dict = pickle.load(open('/Users/dwright/dev/zoo/data/supernova-hunters-vote-fraction-dict.pkl', 'rb'))
  for diff in diff_ids:
    if vote_fraction_dict[diff]<=0.25:
      print(diff)
    labels.append(vote_fraction_dict[diff] > 0.5)
  exit()
  return np.array(labels, dtype='int')

def get_data_vote_fractions(n):
  for i in range(1,n+1):
    print(i)
    data = sio.loadmat('../../../../data/snhunters/3pi_20x20_supernova_hunters_batch_%d_signPreserveNorm_detect_misaligned_images_only.mat'%(i))
    try:
      x_train = np.concatenate((x_train, np.nan_to_num(np.reshape(data['X'], \
        (data['X'].shape[0], 400), order='F'))))
      train_files = np.concatenate((train_files, data['files']))
    except UnboundLocalError:
      x_train = np.nan_to_num(np.reshape(data['X'], (data['X'].shape[0], 400), order='F'))
      train_files = data['files']
  train_diff_ids = files_to_diff_ids(train_files)
  y_train = get_vote_fraction_labels(train_diff_ids)
  """
  for i in range(n+1, n+2):
    print(i)
    data = sio.loadmat('../../../../data/snhunters/3pi_20x20_supernova_hunters_batch_%d_signPreserveNorm_detect_misaligned_images_only.mat'%(i))
    try:
      x_valid = np.concatenate((x_valid, np.nan_to_num(np.reshape(data['X'], \
        (data['X'].shape[0], 400), order='F'))))
      valid_files = np.concatenate((valid_files, data['files']))
    except UnboundLocalError:
      x_valid = np.nan_to_num(np.reshape(data['X'], (data['X'].shape[0], 400), order='F'))
      valid_files = data['files']
  valid_diff_ids = files_to_diff_ids(valid_files)
  y_valid = get_vote_fraction_labels(valid_diff_ids)

  for i in range(n+2,n+3):
    print(i)
    data = sio.loadmat('../../../../data/snhunters/3pi_20x20_supernova_hunters_batch_%d_signPreserveNorm_detect_misaligned_images_only.mat'%(i))
    try:
      x_test = np.concatenate((x_test, np.nan_to_num(np.reshape(data['X'], \
        (data['X'].shape[0], 400), order='F'))))
      test_files = np.concatenate((test_files, data['files']))
    except UnboundLocalError:
      x_test = np.nan_to_num(np.reshape(data['X'], (data['X'].shape[0], 400), order='F'))
      test_files = data['files']
  test_diff_ids = files_to_diff_ids(test_files)
  y_test = get_vote_fraction_labels(test_diff_ids)
  """
  m = y_train.shape[0]
  order = np.random.permutation(m)
  x_train = x_train[order]
  y_train = y_train[order]
  split1 = int(.9*m)
  split2 = int(.95*m)

  return x_train[:split1], y_train[:split1], \
         x_train[split1:split2], y_train[split1:split2], \
         x_train[split2:], y_train[split2:]

def multitask_volunteer_classifications_test(n):

  #_, _, _, _, x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(1)
  #data = sio.loadmat('../../../../data/snhunters/zooniverse_test_set_detect_misaligned.mat')
  #m = data['x'].shape[0]
  #x_valid = np.nan_to_num(np.reshape(data['x'], (data['x'].shape[0], 400), order='F'))[:int(.5*m)]
  #y_valid  = np.squeeze(data['y'])[:int(.5*m)]
  """
  for i in range(n+1,n+1+n):
    print(i)
    data = sio.loadmat('../../../../data/snhunters/3pi_20x20_supernova_hunters_batch_%d_signPreserveNorm_detect_misaligned.mat'%(i))
    print(data.keys())
    
    try:
      x_valid = np.concatenate((x_valid, np.nan_to_num(np.reshape(data['X'], \
        (data['X'].shape[0], 400), order='F'))))
      y_valid = np.concatenate((y_valid, np.squeeze(data['y'])))
    except UnboundLocalError:
      x_valid = np.nan_to_num(np.reshape(data['X'], (data['X'].shape[0], 400), \
        order='F'))
      y_valid = np.squeeze(data['y'])
  
  u, indices = np.unique(x_valid, return_index=True, axis=0)
  x_valid = x_valid[indices]
  y_valid = y_valid[indices]
  """
  """
  data = sio.loadmat('../../../../data/snhunters/3pi_20x20_skew2_signPreserveNorm_volunteer_votes_first_only.mat')
  y_train  = np.squeeze(data['y']) == 'Yes'                                                                                       
  y_test  = np.squeeze(data['y_test']) == 'Yes'                                                                                   
  data = sio.loadmat('../../../../data/snhunters/3pi_20x20_skew2_signPreserveNorm.mat')                                           
  x_train = data['X']                                                                                                             
  x_test = data['testX']                                                                                                          
  # split the data into training, validation and test sets                                                                        
  print(x_train.shape)                                                                                                            
                                                                                                                                  
  order = np.random.permutation(x_train.shape[0])                                                                                 
  x_train = x_train[order]                                                                                                        
  y_train = y_train[order]                                                                                                        
                                                                                                                                  
  split = int(x_train.shape[0]*.75)                                                                                               
  x_valid = x_train[split:]                                                                                                       
  y_valid = y_train[split:]                                                                                                       
  x_train = x_train[:split]                                                                                                       
  y_train = y_train[:split]                                                                                                       
                                                                                                                                  
  print(x_valid.shape)                                                                                                            
  print(x_train.shape)
  
  #data = sio.loadmat('../../../../data/snhunters/3pi_20x20_skew2_signPreserveNorm.mat')
  #x_valid  = data['testX']
  #y_valid  = np.squeeze(data['testy'])
  #x_train  = data['X']
  #y_train  = np.squeeze(data['y'])
  print(x_valid.shape)
  """
  # load the pretrained DEC model for Supernova Hunters
  ae_weights  = '../../../../DEC-keras/results/snhunters/ae_weights.h5'
  #ae_weights = './ae_weights_snh_multitask.h5'
  #dec_weights = '../../../../DEC-keras/results/snh/%d/DEC_model_final.h5'%n_clusters
  #dec_weights = None
  #dec_weights = './results/dec/DEC_model_final.h5'
  dec_weights = '../../../../DEC-keras/results/snhunters/10/DEC_model_final.h5'
  #dec = load_dec(x_valid, ae_weights, dec_weights, n_clusters, batch_size, lr, momentum)

  #dec.clustering(x_valid, np_utils.to_categorical(y_valid), pretrained_weights=dec_weights)

  #exit()

  """
  for i in range(1,n+1):
    print(i)
    data = sio.loadmat('../../../../data/snhunters/3pi_20x20_supernova_hunters_batch_%d_signPreserveNorm_detect_misaligned.mat'%(i))
    try:
      x_train = np.concatenate((x_train, np.nan_to_num(np.reshape(data['X'], \
        (data['X'].shape[0], 400), order='F'))))
      y_train = np.concatenate((y_train, np.squeeze(data['y'])))
    except UnboundLocalError:
      x_train = np.nan_to_num(np.reshape(data['X'], (data['X'].shape[0], 400), \
        order='F'))
      y_train = np.squeeze(data['y'])
  
  u, indices = np.unique(x_train, return_index=True, axis=0)
  x_train = x_train[indices]
  y_train = y_train[indices]
  """
  x_train, y_train, x_valid, y_valid, x_test, y_test = get_data_vote_fractions(n)
  print(x_train.shape)
  #dec = MultitaskLabelsByClusterDEC(dims=[x_valid.shape[-1], 500, 500, 2000, 10], \
  #  n_clusters=n_clusters, batch_size=batch_size)
  dec = MultitaskDEC(dims=[x_valid.shape[-1], 500, 500, 2000, 10], \
    n_clusters=n_clusters, batch_size=batch_size)
  dec.initialize_model(optimizer=SGD(lr=lr, momentum=momentum),
                     ae_weights=ae_weights,
                     x=x_train)
  dec.model.load_weights(dec_weights)
  '''
  cluster_centres = get_cluster_centres(dec)
  labels = [str(i) for i in range(n_clusters)]
  lcolours = ['#CAA8F5', '#D6FF79', '#A09BE7', '#5F00BA', '#56CBF9', \
              '#F3C969', '#ED254E', '#B0FF92', '#D9F0FF', '#46351D']
  pca_plot(dec.encoder, x_train, cluster_centres, y=y_train, labels=labels, lcolours=[lcolours[0], lcolours[1]])
  '''
  #dec.clustering(x_train, np_utils.to_categorical(y_train), \
  #  (x_valid, np_utils.to_categorical(y_valid)), \
  #  pretrained_weights=dec_weights, maxiter=maxiter)

  y_pred, metrics_dict = dec.clustering(x_train, np_utils.to_categorical(y_train), \
                                      (x_valid, np_utils.to_categorical(y_valid)), \
                                      pretrained_weights=dec_weights, maxiter=1000, \
                                      alpha=K.variable(1.0), beta=K.variable(0.0), gamma=K.variable(1.0),  \
                                      loss_weight_decay=True, update_interval=1)
  #dec.clustering(x_train, np_utils.to_categorical(y_train), \
  #  x_valid, np_utils.to_categorical(y_valid), \
  #  pretrained_weights=dec_weights, maxiter=maxiter)
  
  y_pred = dec.model.predict(x_valid)[0]
  f, _, _, _ = one_percent_fpr(y_valid, y_pred[:,1], 0.01)
  print(f)
  print(f1_score(y_valid, np.argmax(y_pred, axis=1)))
  print(np.round(homogeneity_score(y_valid, dec.model.predict(x_valid, verbose=0)[1].argmax(1)), 5))

  
  cluster_centres = get_cluster_centres(dec)
  labels = [str(i) for i in range(n_clusters)]
  lcolours = ['#CAA8F5', '#D6FF79', '#A09BE7', '#5F00BA', '#56CBF9', \
              '#F3C969', '#ED254E', '#B0FF92', '#D9F0FF', '#46351D']
  pca_plot(dec.encoder, x_train, cluster_centres, y=y_train, labels=labels, lcolours=[lcolours[0], lcolours[1]])

  cluster_centres = get_cluster_centres(dec)
  labels = [str(i) for i in range(n_clusters)]
  lcolours = ['#CAA8F5', '#D6FF79', '#A09BE7', '#5F00BA', '#56CBF9', \
              '#F3C969', '#ED254E', '#B0FF92', '#D9F0FF', '#46351D']
  pca_plot(dec.encoder, x_valid, cluster_centres, y=y_valid, labels=labels, lcolours=[lcolours[0], lcolours[1]])

def main():
  multitask_volunteer_classifications_test(23)

if __name__ == '__main__':
  main()
