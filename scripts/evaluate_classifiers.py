import sys, numpy as np, matplotlib.pyplot as plt, seaborn as sns, pickle, os
from sklearn.metrics import roc_curve as roc
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import average_precision_score as aps
import fullstack_ord, fullstack


#Use this function to retrieve class, score data for an ordinal regression model
def auc_calc_scores(model, data, class3only=True):
  #In general we are using the model to predict which sequences from the last sort,
  #i.e. RH03, are strong binders (there is no point predicting which sequences
  #from RH01 or RH02 will be strong binders because sequences in those sorts
  #are weak by definition).
  if class3only == True:
    xprecutoff = data[np.argwhere(data[:,184]>=2).flatten(),:]
  else:
    xprecutoff = np.copy(data)
  scores = model.extract_hidden_rep(xprecutoff).numpy().flatten()
  true_class = np.zeros((xprecutoff.shape[0]))
  true_class[np.argwhere(xprecutoff[:,183]==3).flatten()] = 1
  return true_class, scores


#Function for loading the fulldataset to evaluate a model.
def retrieve_full_dataset(separate_train_test = False):
  current_dir = os.getcwd()
  os.chdir(os.path.join('..', 'processed_data'))
  ord_data1 = np.load('9site_train.npy')
  ord_data2 = np.load('9site_test.npy')

  os.chdir(current_dir)
  if separate_train_test == False:
    ord_data = np.vstack([ord_data1, ord_data2])
    return ord_data
  else:
    return (ord_data1, ord_data2)

#This function retrieves the subsampled datasets generated earlier
#by the sequence encoding module. These can be used to evaluate reproducibility
#(how much does ROC-AUC change on repeated retraining of the model
#using a smaller dataset to train it)?
def retrieve_subsampled_dataset():
  current_dir = os.getcwd()
  os.chdir(os.path.join('..', 'processed_data', 'twenty_percent_subsampling'))
  subsample_twenty_ord = [np.load('subsample_ord_%s.npy'%i) for i in range(0,5)]
  os.chdir(current_dir)
  os.chdir(os.path.join('..', 'processed_data', 'ten_percent_subsampling'))
  subsample_ten_ord = [np.load('subsample_ord_%s.npy'%i) for i in range(0,5)]
  os.chdir(current_dir)
  return subsample_twenty_ord, subsample_ten_ord

#Calculate AUC-ROC for RH04 sequences in RH03 for both the training and
#test sets.
def calc_RH04_auc_train_test():
  current_dir = os.getcwd()
  os.chdir(os.path.join('..', 'models'))
  with open('final_model', 'rb') as model_file:
    ord_model = pickle.load(model_file)

  os.chdir(current_dir)
  ord_data = retrieve_full_dataset(separate_train_test=True)

  ytrue_ord_train, yscores_ord_train = auc_calc_scores(ord_model, ord_data[0])
  print('Training set ordinal AUC: %s'%auc(ytrue_ord_train, yscores_ord_train))

  ytrue_ord_test, yscores_ord_test = auc_calc_scores(ord_model, ord_data[1])
  print('Test set ordinal AUC: %s'%auc(ytrue_ord_test, yscores_ord_test))  


#Plot the score distributions.
def plot_score_distributions():
  current_dir = os.getcwd()
  os.chdir(os.path.join('..', 'models'))
  with open('final_model', 'rb') as model_file:
    ord_model = pickle.load(model_file)

  os.chdir(current_dir)
  ord_data = retrieve_full_dataset()
  y_ord = ord_data[:,184]
  class3only_ord = np.copy(y_ord)
  class3only_ord[np.argwhere(class3only_ord==3).flatten()]=2
  y_ord[np.argwhere(ord_data[:,183]==3)]=3


  #We are using a kernel density estimate or kdeplot from seaborn. The y-axis will
  #only indicate density relative to the rest of the distribution. We create two
  #plots, one with sequences that are RH04 marked and the other with them unmarked.
  fig, (ax1, ax2) = plt.subplots(2,1)
  scores = ord_model.extract_hidden_rep(ord_data).numpy().flatten()
  sns.kdeplot(scores[np.argwhere(class3only_ord==0).flatten()],shade=True,color='blue',bw=0.1,ax=ax1)
  sns.kdeplot(scores[np.argwhere(class3only_ord==1).flatten()],shade=True,color='orange',bw=0.1,ax=ax1)
  sns.kdeplot(scores[np.argwhere(class3only_ord==2).flatten()],shade=True,color='green',bw=0.1,ax=ax1)
  ax1.legend(['RH01', 'RH02', 'RH03', 'RH04'])
  ax1.set_xlabel('Score distribution')
  ax1.set_ylabel('Kernel density estimate')
  ax1.set_title('KDE-smoothed score distribution using ordinal regression,\nRH04 sequences unmarked')

  sns.kdeplot(scores[np.argwhere(y_ord==0).flatten()],shade=True,color='blue',bw=0.1,ax=ax2)
  sns.kdeplot(scores[np.argwhere(y_ord==1).flatten()],shade=True,color='orange',bw=0.1,ax=ax2)
  sns.kdeplot(scores[np.argwhere(y_ord==2).flatten()],shade=True,color='green',bw=0.1,ax=ax2)
  sns.kdeplot(scores[np.argwhere(y_ord==3).flatten()],shade=True,color='red',bw=0.1,ax=ax2)
  ax2.legend(['RH01', 'RH02', 'RH03', 'RH04'])
  ax2.set_xlabel('Score distribution')
  ax2.set_ylabel('Kernel density estimate')
  ax2.set_title('KDE-smoothed score distribution using ordinal regression,\nRH04 marked')
  plt.show()



  #Now calculate ROC-AUC for the model for RH04 predictions.
  ytrue_ord, yscores_ord = auc_calc_scores(ord_model, ord_data)
  print('Ordinal AUC: %s'%auc(ytrue_ord, yscores_ord))
  fpr_ord, tpr_ord, _ = roc(ytrue_ord, yscores_ord)


  equal_spec = np.arange(1,-0.1,-0.1)

  plt.plot(fpr_ord, tpr_ord, color='blue')
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('AUC-ROC for RH04 prediction based on an ordinal regression\n model trained on RH01-RH03 data only, 9-site library')
  plt.plot(equal_spec, equal_spec, color='black', linestyle='--')
  plt.show()


#This function retrieves the subsampled datasets (five subsampled at 20%, five subsampled
#at 10%) and trains an ordinal regression model on
#each of the subsampled datasets, then uses the resulting model to generate scores
#for the FULL dataset and calculate AUC-ROC for prediction of RH04 sequences among
#RH03 to evaluate performance.
def evaluate_subsampling(use_weights=True):
  twenty_ord, ten_ord = retrieve_subsampled_dataset()
  full_ord_data = retrieve_full_dataset()
  #Use same dropout and number of epochs for all to stay consistent.
  dropout, epochs, aucs = 0.3, 20, []
  current_dir = os.getcwd()
  os.chdir('..')
  output_file = open('subsampling AUC-ROC results.txt', 'w+')
  output_file.write('Use weights = %s\n'%use_weights)
  for i in range(0,5):
    model = fullstack_ord.NN(dropout=dropout, input_dim=180, l2=0.000)
    losses = model.trainmod(twenty_ord[i],epochs=epochs, minibatch=250,
                         lr=0.005, use_weights=use_weights)
    ytrue, yscores = auc_calc_scores(model, full_ord_data)
    aucs.append(auc(ytrue, yscores))
  output_file.write('Ordinal regression, 20 percent subsampling: AUC %s +/- %s\n'%(np.mean(aucs),
                                                                     np.std(aucs)))
  aucs = []
  for i in range(0,5):
    model = fullstack_ord.NN(dropout=dropout, input_dim=180, l2=0.000)
    losses = model.trainmod(ten_ord[i],epochs=epochs, minibatch=250,
                         lr=0.005, use_weights=use_weights)
    ytrue, yscores = auc_calc_scores(model, full_ord_data)
    aucs.append(auc(ytrue, yscores))
  output_file.write('Ordinal regression, 10 percent subsampling: AUC %s +/- %s'%(np.mean(aucs),
                                                                     np.std(aucs)))
  os.chdir(current_dir)
  output_file.close()
  
