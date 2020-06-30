import sys, numpy as np, matplotlib.pyplot as plt, seaborn as sns, pickle, os
from sklearn.metrics import roc_curve as roc
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import average_precision_score as aps
from train_models import load_data, build_models, filter_bin_data


#This function retrieves the subsampled datasets generated earlier
#by the sequence encoding module. These can be used to evaluate reproducibility
#(how much does ROC-AUC change on repeated retraining of the model
#using a smaller dataset to train it) for the ordinal regression model.
def retrieve_subsampled_dataset():
    current_dir = os.getcwd()
    os.chdir(os.path.join('..', 'processed_data', 'twenty_percent_subsampling'))
    x20_ord = [np.load('subsample_ord_%s.npy'%i) for i in range(0,5)]
    x20_enrich = [np.load('subsample_enrich_%s.npy'%i) for i in range(5)]
    x20_nom = []
    for i in range(0,5):
        x = np.load("subsample_nom_%s.npy"%i)
        x20_nom.append(filter_bin_data(x))
    os.chdir(current_dir)
    
    os.chdir(os.path.join('..', 'processed_data', 'ten_percent_subsampling'))
    x10_ord = [np.load('subsample_ord_%s.npy'%i) for i in range(0,5)]
    x10_nom = []
    x10_enrich = [np.load('subsample_enrich_%s.npy'%i) for i in range(5)]
    for i in range(0,5):
        x = np.load("subsample_nom_%s.npy"%i)
        x10_nom.append(filter_bin_data(x))
    os.chdir(current_dir)
    return x20_ord, x10_ord, x20_nom, x10_nom, x20_enrich, x10_enrich

#Calculate AUC-ROC for RH04 sequences in RH03 for both the training and
#test sets.
def calc_RH04_auc_train_test(rh03_only=False):
    current_dir = os.getcwd()
    os.chdir(os.path.join('..', 'init_models'))
    models = dict()
    for mod_type in ("ord", "enrich", "bin", "rf"):
        with open("%s_model"%mod_type, 'rb') as model_file:
            models[mod_type] = pickle.load(model_file)
    os.chdir(current_dir)
    
    model_names = {"ord":"ordinal regression", "bin":"classification neural network",
                "enrich":"enrichment neural network", "rf":"random forest classifier"}
    colors = ("red", "blue", "green", "orange")
    for i, mod_type in enumerate(("ord", "enrich", "bin", "rf")):
        current_data = load_data(model_type=mod_type, return_validation_set=False,
                            return_rh04=True, rh03_only = rh03_only)
        xtrain, rh04train, xtest, rh04test = current_data[0], current_data[1], current_data[2],\
                                        current_data[3]
        trainpreds, testpreds = gen_model_spec_preds(models[mod_type], mod_type, xtrain, xtest)
        print('Training set %s AUC: %s'%(mod_type, auc(rh04train, trainpreds)))
        print('Test set %s AUC: %s'%(mod_type, auc(rh04test, testpreds)))
        fpr_ord, tpr_ord, _ = roc(rh04test, testpreds)
        equal_spec = np.arange(1,-0.1,-0.1)

        plt.plot(fpr_ord, tpr_ord, color=colors[i], label=model_names[mod_type])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
    if rh03_only == True:
        plt.title('AUC-ROC for RH04 prediction on the held-out test set'
                '\nusing models trained on RH01 - RH03 data only;\nAUC-ROC of ranking for RH03 sequences')
    else:
        plt.title('AUC-ROC for RH04 prediction on the held-out test set'
                '\nusing models trained on RH01 - RH03 data only;\nAUC-ROC of ranking for RH01 - RH03 sequences')

    plt.legend()
    plt.plot(equal_spec, equal_spec, color='black', linestyle='--')
    plt.show()


def gen_model_spec_preds(model, mod_type, xtrain, xtest=None):
    testpreds = None
    if mod_type == "ord":
        trainpreds = model.extract_hidden_rep(xtrain)
        if xtest is not None:
            testpreds = model.extract_hidden_rep(xtest)
    elif mod_type == "bin":
        trainpreds = model.predict(xtrain)[0]
        if xtest is not None:
            testpreds = model.predict(xtest)[0]
    elif mod_type == "rf":
        trainpreds = model.predict_proba(xtrain[:,0:180])[:,1]
        if xtest is not None:
            testpreds = model.predict_proba(xtest[:,0:180])[:,1]
    elif mod_type == "enrich":
        trainpreds = model.predict(xtrain)[1]
        if xtest is not None:
            testpreds = model.predict(xtest)[1]
    return trainpreds, testpreds

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





#This function retrieves the subsampled datasets (five subsampled at 20%, five subsampled
#at 10%) and trains an ordinal regression model on
#each of the subsampled datasets, then uses the resulting model to generate scores
#for the FULL dataset and calculate AUC-ROC for prediction of RH04 sequences among
#RH03 to evaluate performance.
def evaluate_subsampling(use_weights=True):
    x20_ord, x10_ord, x20_nom, x10_nom, x20_enrich, x10_enrich = retrieve_subsampled_dataset()
    xtrain_ord, ytrain_ord, xtest_ord, ytest_ord = load_data(model_type="ord",
                            return_validation_set=False, return_rh04=True,
                            rh03_only=True)
    xtrain_nom, ytrain_nom, xtest_nom, ytest_nom = load_data(model_type="bin",
                            return_validation_set=False, return_rh04=True,
                            rh03_only=True)
    x_ord = np.vstack([xtrain_ord, xtest_ord])
    rh04_ord = np.concatenate([ytrain_ord, ytest_ord])
    x_nom = np.vstack([xtrain_nom, xtest_nom])
    rh04_nom = np.concatenate([ytrain_nom, ytest_nom])
    #Use same dropout and number of epochs for all to stay consistent.
    current_dir = os.getcwd()
    os.chdir('..')
    output_file = open('subsampling AUC-ROC results.txt', 'w+')
    model_scores = {"bin":[], "rf":[], "ord":[], "enrich":[]}
    for i in range(0,5):
        os.chdir(current_dir)
        build_models(x20_ord[i], ytrain=None,
                        model_type="ord", subsampling=True)
        build_models(x20_nom[i], ytrain=None,
                        model_type="bin", subsampling=True)
        ytrain = x20_nom[i][:,180]
        build_models(x20_nom[i], ytrain,
                        model_type="rf", subsampling=True)
        build_models(x20_enrich[i], ytrain=None, model_type="enrich",
                        subsampling=True)
        os.chdir(os.path.join('..',"subsampling_models"))
        for mod_type in ("ord", "bin", "rf", "enrich"):
            with open("current_%s_model"%mod_type, 'rb') as model_file:
                current_model = pickle.load(model_file)
            trainpreds, _ = gen_model_spec_preds(current_model, mod_type,
                                            x_ord)
            model_scores[mod_type].append(auc(rh04_ord, trainpreds))
    for key in model_scores:
        average, std_dev = np.mean(model_scores[key]), np.std(model_scores[key])
        output_file.write("%s model type: 20 percent subsample: AUC %s +/- %s\n"%(key, average, std_dev))
    
    model_scores = {"bin":[], "rf":[], "ord":[], "enrich":[]}
    for i in range(0,5):
        os.chdir(current_dir)
        build_models(x10_ord[i], ytrain=None,
                        model_type="ord", subsampling=True)
        build_models(x10_nom[i], ytrain=None,
                        model_type="bin", subsampling=True)
        ytrain = x10_nom[i][:,180]
        build_models(x10_nom[i], ytrain,
                        model_type="rf", subsampling=True)
        build_models(x10_enrich[i], ytrain=None, model_type="enrich",
                        subsampling=True)
        os.chdir(os.path.join('..',"subsampling_models"))
        for mod_type in ("ord", "bin", "rf", "enrich"):
            with open("current_%s_model"%mod_type, 'rb') as model_file:
                current_model = pickle.load(model_file)
            trainpreds, _ = gen_model_spec_preds(current_model, mod_type,
                                            x_ord)
            model_scores[mod_type].append(auc(rh04_ord, trainpreds))
    for key in model_scores:
        average, std_dev = np.mean(model_scores[key]), np.std(model_scores[key])
        output_file.write("%s model type: 10 percent subsample: AUC %s +/- %s\n"%(key, average, std_dev))
    output_file.close()
    os.chdir(current_dir) 
