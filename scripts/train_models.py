import numpy as np, matplotlib.pyplot as plt, seaborn as sns, pickle, os
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.model_selection import train_test_split
import fullstack_ord, fullstack

#Use the same dropout and number of epochs across models (initial cross-
#validations were used to select)
dropout = 0.3
epochs = 20

#This function will split up the data into training and validation sets
#(leaving aside the preselected test set, which is set aside at the time
#the sequences are encoded). The matthews correlation coefficient for
#validation (and test set if specified to check it) are printed.
def eval_alternative_models(use_weights = True, look_at_test_set = False,
                            check_rf = True):
    current_dir = os.getcwd()
    os.chdir(os.path.join('..', 'processed_data'))
    ord_train = np.load('9site_train.npy')
    nom_train = np.load('9site_nominal_class_train.npy')
    nom_test = np.load('9site_nominal_class_test.npy')
    model = fullstack_ord.NN(dropout=dropout, input_dim=180, l2=0.0000)
    
    xtrain, xval, ytrain, yval = train_test_split(ord_train, ord_train[:,184],
                                                  test_size=0.2)
    losses = model.trainmod(xtrain,epochs=epochs, minibatch=250,
                         lr=0.005, use_weights=use_weights)
        
    print('ordinal regression train-validation results')
    print('Train: %s'%mcc(ytrain, model.predict(xtrain)[1]))
    print('Validation: %s'%mcc(yval, model.predict(xval)[1]))
    if look_at_test_set == True:
        ord_test = np.load('9site_test.npy')
        print('Test: %s'%mcc(ord_test[:,184], model.predict(ord_test)[1]))
    print('ordinal regression results accounting for frequency')
    print('Train: %s'%mcc(ytrain, model.predict(xtrain)[1], sample_weight=xtrain[:,-1]))
    print('Validation: %s'%mcc(yval, model.predict(xval)[1], sample_weight=xval[:,-1]))
    if look_at_test_set == True:
        print('Test: %s'%mcc(ord_test[:,184], model.predict(ord_test)[1], sample_weight=ord_test[:,-1]))
    print('\n\n')

    if check_rf == True:
        if look_at_test_set == False:
            evaluate_rf_classifier(xtrain, ytrain, xval, yval)
        else:
            evaluate_rf_classifier(xtrain, ytrain, xval, yval, ord_test)

    model = fullstack.NN(dropout=dropout,input_dim=180,l2=0)

    os.chdir(current_dir)
    
    xtrain, xval, ytrain, yval = train_test_split(nom_train,
                                                        nom_train[:,180],
                                                        test_size=0.2)
    losses = model.trainmod(xtrain,epochs=epochs, minibatch=250,
                         lr=0.005, use_weights=use_weights)
    print('nominal regression train-validation results')
    print('Train: %s'%mcc(ytrain, model.predict(xtrain)[1]))
    print('Validation: %s'%mcc(yval, model.predict(xval)[1]))
    if look_at_test_set == True:
        print('Test: %s'%mcc(nom_test[:,180], model.predict(nom_test)[1]))
    print('nominal regression results accounting for frequency')
    print('Train: %s'%mcc(ytrain, model.predict(xtrain)[1], sample_weight=xtrain[:,-1]))
    print('Test: %s'%mcc(yval, model.predict(xval)[1], sample_weight=xval[:,-1]))
    if look_at_test_set == True:
        print('Test: %s'%mcc(nom_test[:,180], model.predict(nom_test)[1], sample_weight=nom_test[:,-1]))
    print('\n\n')

#This function recombines test and training sets and trains a model
#on the full dataset, then saves to the models directory.
def build_final_models(use_weights=False):
    train_ord_data = np.load('9site_train.npy')
    test_ord_data = np.load('9site_test.npy')
    ord_data = np.vstack([train_ord_data, test_ord_data])
    model = fullstack_ord.NN(dropout=dropout, input_dim=180, l2=0)
    _ = model.trainmod(ord_data, epochs=epochs, minibatch=250,
                           use_weights=use_weights, lr=0.005)
    current_dir = os.getcwd()
    os.chdir(os.path.join('..', 'models'))
    with open('final_model', 'wb') as output_file:
        pickle.dump(model, output_file)

    os.chdir(current_dir)
    train_ord_data = np.load('9site_nominal_class_train.npy')
    test_ord_data = np.load('9site_nominal_class_test.npy')
    nom_data = np.vstack([train_nom_data, test_nom_data])
    model = fullstack.NN(dropout=dropout, input_dim=180, l2=0)
    _ = model.trainmod(nom_data, epochs=epochs, minibatch=250,
                           use_weights=use_weights, lr=0.005)
    os.chdir(os.path.join('..', 'models'))
    with open('nominal_data_model', 'wb') as output_file:
        pickle.dump(model, output_file)
    os.chdir(current_dir)


#This function is used to evaluate the random forest classifier as
#a possible alternative to fully connected networks / ordinal regression.
def evaluate_rf_classifier(xtrain, ytrain, xval, yval, test_set=None):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=500, n_jobs=3,
                                       min_samples_split=100, min_samples_leaf=5, oob_score=True)
    model.fit(xtrain[:,0:180], ytrain, sample_weight=xtrain[:,-1])
    print('OOB Accuracy: %s'%model.oob_score_)
    trainscore = mcc(model.predict(xtrain[:,0:180]), ytrain)
    testscore = mcc(model.predict(xval[:,0:180]), yval)
    print('Training set MCC for random forest: %s'%trainscore)
    print('Validation set MCC for random forest: %s'%testscore)
    if test_set is not None:
        testscore = mcc(model.predict(test_set[:,0:180]), test_set[:,184])
        print('Test set MCC for random forest: %s'%testscore)

    trainscore = mcc(model.predict(xtrain[:,0:180]), ytrain, sample_weight=xtrain[:,-1])
    testscore = mcc(model.predict(xval[:,0:180]), yval, sample_weight=xval[:,-1])
    print('Training set MCC for random forest (frequency weighted): %s'%trainscore)
    print('Validation set MCC for random forest (frequency weighted): %s'%testscore)
    if test_set is not None:
        testscore = mcc(model.predict(test_set[:,0:180]), test_set[:,184], sample_weight=test_set[:,-1])
        print('Test set MCC for random forest (frequency weighted): %s'%testscore)
    
