import numpy as np, matplotlib.pyplot as plt, seaborn as sns, pickle, os
from time import sleep
from sklearn.metrics import matthews_corrcoef as mcc, r2_score
from sklearn.model_selection import train_test_split



from neural_network_classes.fullstack_ord import ord_nn
from neural_network_classes.fullstack import nominal_classifier
from neural_network_classes.binary_classifier import bin_class_nn
from neural_network_classes.enrichment_pred_nn import enrichment_nn
from sklearn.ensemble import RandomForestClassifier




#UNFORTUNATELY because of the need to compare a variety of different models with different criteria for each,
#the data has been encoded in a variety of different ways -- if we started over, we could clean this up quite a bit,
#but this scheme evolved over the course of the project. For now we use this function to load the dataset
#appropriate to any one given model and either keep the training set together or split it up into training and validation,
#as appropriate. We also offer the option to return instead of the category info usually returned
#as ytrain or ytest a binary indicator of whether or not the sequence is present in RH04.
def load_data(model_type = "ord", return_validation_set = True,
                return_rh04 = False, rh03_only=False):
    current_dir = os.getcwd()
    os.chdir(os.path.join('..', 'processed_data'))

    if model_type == "ord":
        train_data = np.load('9site_train.npy')
        xtest = np.load('9site_test.npy')
        if rh03_only == True:
            train_data = train_data[train_data[:,184]==2]
            xtest = xtest[xtest[:,184]==2]
        xtrain, xval, ytrain, yval = train_test_split(train_data, train_data[:,184],
                                                  test_size=0.2)
        ytest = xtest[:,184]
    elif model_type == "enrich":
        train_data = np.load('9site_enrich_train.npy')
        xtest = np.load('9site_enrich_test.npy')
        xtrain, xval, ytrain, yval = train_test_split(train_data, train_data[:,180],
                                                  test_size=0.2)
        ytest = xtest[:,180]
    elif model_type == "bin" or model_type == "rf":
        #FOr the binary classifiers, we train on RH02 vs RH03 only because...they're
        #binary classifiers.
        train_data = np.load('9site_nominal_class_train.npy')
        xtest = np.load('9site_nominal_class_test.npy')
        if rh03_only == True and return_rh04 == True:
            train_data = filter_bin_data(train_data, True)
            xtest = filter_bin_data(xtest, True)
        else:
            train_data = filter_bin_data(train_data)
            xtest = filter_bin_data(xtest)
        xtrain, xval, ytrain, yval = train_test_split(train_data, train_data[:,180],
                                                  test_size=0.2)
        ytest = xtest[:,180]
    else:
        train_data = np.load('9site_nominal_class_train.npy')
        xtest = np.load('9site_nominal_class_test.npy')
        xtrain, xval, ytrain, yval = train_test_split(train_data, train_data[:,180],
                                                  test_size=0.2)
        ytest = xtest[:,180] 
    os.chdir(current_dir)

    if return_validation_set:
        return xtrain, xval, ytrain, yval, xtest, ytest
    elif return_rh04:
        xtrain = np.vstack([xtrain, xval])
        #If the caller asks for rh04 info we'll return that instead of ytrain and ytest.
        #If so, they do not need a validation set because that action is only taken at
        #the testing stage.
        if model_type == "ord":
            rh04_train = xtrain[:,183]
            rh04_test = xtest[:,183]
        else:
            rh04_train = xtrain[:,181]
            rh04_test = xtest[:,181]
        rh04_train[rh04_train>=1] = 1
        rh04_test[rh04_test>=1] = 1
        return xtrain, rh04_train, xtest, rh04_test
    else:
        xtrain = np.vstack([xtrain, xval])
        ytrain = np.concatenate([ytrain, yval])
        return xtrain, ytrain, xtest, ytest


def filter_bin_data(x, keep_rh03_only=False):
    if keep_rh03_only == True:
        x = x[x[:,180]==2]
        return x
    x = x[x[:,180]>0]
    y = np.copy(x[:,180])
    y[y==1] = 0
    y[y==2] = 1
    x[:,180] = y
    return x

#THis function prints the results of model evaluation for any model passed into eval_model.
def print_model_eval_results(train_mcc, val_mcc, test_mcc=None, model_description=''):
    print("Training set MCC for %s: %s"%(model_description, train_mcc))
    print("Validation set MCC for %s: %s"%(model_description, val_mcc))
    if test_mcc is not None:
        print("Test set MCC for %s: %s"%(model_description, test_mcc))

#Evaluate a model on a training and validation set (or, if specified, on a test set).
#The test set is only used at the end and is held out from all initial evaluations.
def eval_model(model_type = "ord", look_at_test_set=False, dropout=0):
    #Use the same dropout and number of epochs across models (initial cross-
    #validations were used to select)
    epochs = 40
    testscore = None
    xtrain, xval, ytrain, yval, xtest, ytest = load_data(model_type)
    if model_type == "rf":
        if look_at_test_set == True:
            model = RandomForestClassifier(n_estimators=1000, n_jobs=3,
                                       min_samples_split=35, min_samples_leaf=5, oob_score=True)
            xtrain = np.vstack([xtrain, xval])
            ytrain = np.concatenate([ytrain, yval])
            model.fit(xtrain[:,0:180], ytrain, sample_weight=xtrain[:,-1])
            trainpreds, valpreds, testpreds = model.predict(xtrain[:,0:180]), model.predict(xval[:,0:180]),\
                                          model.predict(xtest[:,0:180])
        else:
            model = RandomForestClassifier(n_estimators=1000, n_jobs=3,
                                       min_samples_split=35, min_samples_leaf=5, oob_score=True)
            model.fit(xtrain[:,0:180], ytrain, sample_weight=xtrain[:,-1])
            trainpreds, valpreds, testpreds = model.predict(xtrain[:,0:180]), model.predict(xval[:,0:180]),\
                                          model.predict(xtest[:,0:180])
    elif model_type == "enrich":
        model = enrichment_nn(dropout=dropout, input_dim=180, l2=0.0000)
        model.trainmod(xtrain,epochs=epochs, minibatch=250, lr=0.005, use_weights=True)
        trainpreds, valpreds, testpreds = model.predict(xtrain[:,0:180])[1], model.predict(xval[:,0:180])[1],\
                                          model.predict(xtest[:,0:180])[1]
        trainscore = r2_score(trainpreds, ytrain)
        valscore = r2_score(valpreds, yval)
        if look_at_test_set == True:
            testscore = r2_score(testpreds, ytest)
    else:
        model_dict = {"ord":ord_nn, "nom":nominal_classifier, "bin":bin_class_nn}
        dropout_dict = {"ord":0.3, "nom":0.3, "bin":0.4}
        model = model_dict[model_type](dropout=dropout_dict[model_type], input_dim=180, l2=0.0000)
        if model_type == "bin" and look_at_test_set == True:
            xtrain = np.vstack([xtrain, xval])
            ytrain = np.concatenate([ytrain, yval])
        model.trainmod(xtrain,epochs=epochs, minibatch=250, lr=0.005, use_weights=True)
        trainpreds, valpreds, testpreds = model.predict(xtrain[:,0:180])[1], model.predict(xval[:,0:180])[1],\
                                          model.predict(xtest[:,0:180])[1]
    if model_type != "enrich":
        trainscore = mcc(trainpreds, ytrain)
        valscore = mcc(valpreds, yval)
        if look_at_test_set == True:
            testscore = mcc(testpreds, ytest)
    print_model_eval_results(trainscore, valscore, testscore, model_description = model_type)




#This function trains models either on the training set (prior to final model evaluation)
#or on the entire dataset (if we are building a model to make predictions about 
#sequences not present in any sort we've seen so far and want our training set to be as large
#as possible). This last step is ONLY carried out after a model has been selected and
#evaluated on the test set.
def build_models(xtrain, ytrain, xtest=None, ytest=None,
                model_type="ord", final=False, use_weights=True, dropout=0,
                subsampling=False):
    if final == True:
        epochs=40
    else:
        epochs=30
    if final == True:
        xtrain = np.vstack([xtrain, xtest])
        ytrain = np.concatenate([ytrain, ytest])
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=500, n_jobs=3,
                                       min_samples_split=100, min_samples_leaf=5, oob_score=True)
        model.fit(xtrain[:,0:180], ytrain, sample_weight=xtrain[:,-1])
    elif model_type == "enrich":
        model = enrichment_nn(dropout=dropout, input_dim=180, l2=0.0000)
        model.trainmod(xtrain,epochs=epochs, minibatch=250, lr=0.005, use_weights=True)
    else:
        dropout_dict = {"ord":0.3, "nom":0.3, "bin":0.3}
        model_dict = {"ord":ord_nn, "nom":nominal_classifier, "bin":bin_class_nn}
        model = model_dict[model_type](dropout=dropout_dict[model_type], input_dim=180, l2=0.0000)
        model.trainmod(xtrain,epochs=epochs, minibatch=250, lr=0.005, use_weights=True)

    current_dir = os.getcwd()
    if subsampling == True:
        os.chdir(os.path.join('..', 'subsampling_models'))
        with open("current_%s_model"%model_type, 'wb') as output_file:
            pickle.dump(model, output_file)
    elif final == True:
        os.chdir(os.path.join('..', 'final_models'))
        with open('final_%s_model'%model_type, 'wb') as output_file:
            pickle.dump(model, output_file)
    else:
        os.chdir(os.path.join('..', 'init_models'))
        with open('%s_model'%model_type, 'wb') as output_file:
            pickle.dump(model, output_file)
    os.chdir(current_dir)
