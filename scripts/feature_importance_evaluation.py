import numpy as np, matplotlib.pyplot as plt, seaborn as sns, sys, pickle, os
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.model_selection import train_test_split

#Global, since we will use this throughout.
aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
position_list = [0,2,5,29,36,37,39,40,42]

#Function for loading the fulldataset. We will use this to train a contextual
#regression model and then generate feature importance scores.
def retrieve_full_dataset():
    current_dir = os.getcwd()
    os.chdir(os.path.join('..', 'processed_data'))
    ord_data1 = np.load('9site_train.npy')
    ord_data2 = np.load('9site_test.npy')
    ord_data = np.vstack([ord_data1, ord_data2])
    os.chdir(current_dir)
    return ord_data

def train_contextual_regression():
    ord_data = retrieve_full_dataset()

    #We need to use a higher level of dropout for the contextual regression
    #to get MCC roughly equivalent to what we get from ordinal regression /
    #nominal classification.
    dropout = 0.5
    epochs = 20
    model = fullstack_contextualord.NN(dropout = dropout)
    model.trainmod(ord_data, epochs=epochs, minibatch=250, use_weights=True, lr=0.005)
    current_dir = os.getcwd()
    os.chdir(os.path.join('..', 'models'))
    with open('contextual_regression_model', 'wb') as model_file:
        pickle.dump(model, model_file)
    os.chdir(current_dir)

def score_known_high_binders():
    ord_data = retrieve_full_dataset()
    current_dir = os.getcwd()
    os.chdir(os.path.join('..', 'final_models'))
    with open('final_model', 'rb') as model_file:
        ord_model = pickle.load(model_file)

    with open('final_bin_model', 'rb') as model_file:
        bin_model = pickle.load(model_file)

    with open('final_rf_model', 'rb') as model_file:
        rf_model = pickle.load(model_file)

    with open('final_enrich_model', 'rb') as model_file:
        enrich_model = pickle.load(model_file)
    os.chdir(current_dir)
    os.chdir(os.path.join('..', 'processed_data'))
    known_high_binders = np.load('known_high_binders_9site.npy')
    os.chdir(current_dir)
    fullset_scores = ord_model.extract_hidden_rep(ord_data)
    scores = ord_model.extract_hidden_rep(known_high_binders)
    calc_print_percentiles(fullset_scores, scores, "Ordinal reg")
    
    fullset_scores = enrich_model.predict(ord_data)[1]
    scores = enrich_model.predict(known_high_binders)[1]
    calc_print_percentiles(fullset_scores, scores, "Enrichment")
    
    fullset_scores = rf_model.predict_proba(ord_data[:,0:180])[:,1]
    scores = rf_model.predict_proba(known_high_binders[:,0:180])[:,1]
    calc_print_percentiles(fullset_scores, scores, "Random forest")

    fullset_scores = bin_model.predict(ord_data)[0]
    scores = bin_model.predict(known_high_binders)[0]
    calc_print_percentiles(fullset_scores, scores, "Bin NN")

def calc_print_percentiles(all_scores, highbinder_scores, model_type):
    print('%s scores for known_high_binders:'%model_type)
    print(highbinder_scores)
    print("Percentiles:")
    percentiles = [100*all_scores[all_scores<highbinder_scores[i]].shape[0] / 
                    all_scores.shape[0] for i in range(0, highbinder_scores.
                        shape[0])]
    print(percentiles)
    print('\n')


def gen_high_scoring_seq_feature_importance(ord_model, ord_data):
    current_dir = os.getcwd()
    os.chdir(os.path.join('..', 'models'))
    with open('contextual_regression_model', 'rb') as model_file:
        context_model = pickle.load(model_file)
    os.chdir(current_dir)
    #Score the dataset using the ordinal regression model. Take the 50 highest scoring sequences
    #and generate feature importances for each using the contextual regression model, then
    #average them.
    dataset_scores = ord_model.extract_hidden_rep(ord_data).numpy().flatten()
    highscores = np.argsort(dataset_scores)[-50:]
    best_seqs = ord_data[highscores,:]
    feat_importances_cr = context_model.generate_feat_importances(best_seqs)
    feat_importances_cr = np.mean(feat_importances_cr,0)
    #Create a plot of the averaged feature importances.
    fig, axes = plt.subplots(2,5, sharey=True)
    for i in range(0,2):
        for j in range(0,5):
            if j >= 4 and i == 1:
                        break
            _ = axes[i][j].bar(np.arange(20), feat_importances_cr[((i*5+j)*20):((i*5+j)*20+20)])
            _ = axes[i][j].set_xticks(np.arange(20))
            _ = axes[i][j].set_xticklabels(aas)
            _ = axes[i][j].set_xlabel('Amino acids at position %s'%(position_list[(i*5)+j]+1))
                
    plt.subplots_adjust(left=0.05, right=0.99)
    plt.suptitle('Feature importance scores per position using a contextual regression model')
    plt.show()
    #Identify the three BEST bets at each position based on the averaged feature importances
    #for the top 50 sequences. Return these so we can use them to generate "best bet" sequences
    #and score them.
    best_positions = []
    for i in range(0,9):
        current_subset = feat_importances_cr[(20*i):(20*i+20)]
        best_positions.append(np.argsort(current_subset)[-3:].tolist())
    return best_positions

#This function will generate the "best bet" sequences using the top three amino acids at each position
#based on contextual regression, then score all of these using the ordinal regression model and print
#out the top two.
def find_top_two_seqs():
    ord_data = retrieve_full_dataset()
    current_dir = os.getcwd()
    os.chdir(os.path.join('..', 'models'))
    with open('final_model', 'rb') as model_file:
        ord_model = pickle.load(model_file)
    with open('contextual_regression_model', 'rb') as model_file:
        context_model = pickle.load(model_file)
    os.chdir(current_dir)
    best_positions = gen_high_scoring_seq_feature_importance(ord_model, ord_data)

    seqs_of_interest = np.stack(gen_candidate_seqs(best_positions))

    pred_scores = ord_model.extract_hidden_rep(seqs_of_interest).numpy().flatten()
    sorted_scores = np.argsort(pred_scores)[-2:]
    top_scorers = seqs_of_interest[sorted_scores,:]
    best_mutants = []
    for i in range(0,2):
        current_aa_sequence = []
        for j in range(0,9):
            current_aa_sequence.append(aas[np.argmax(top_scorers[i,(j*20):(j*20+20)])])
        current_aa_sequence = ''.join(current_aa_sequence)
        best_mutants.append(current_aa_sequence)
        print(current_aa_sequence)

    raw_temp = 'gaatatgtggtggaaaaagttctcgaccgtcgagtggtaaagggcaaagtggagtacctcctaaagtggaagggattctcagatgaggacaacacatgggagccagaagagaacctggattgccccgacctcattgctgagtttctgcagtcacagaaaaca'
    aa_template = 'EYVVEKVLDRRVVKGKVEYLLKWKGFSDEDNTWEPEENLDCPDLIAEFLQSQKT' 
    raw_temp = raw_temp.upper()

    codon_dict = {'W':'TGG', 'E':'GAA', 'D':'GAT', 'V':'GTT', 'N':'AAC', 'K':'AAA', 'C':'TGT', 'A':'GCA'}
    nuc_seqs = []
    for i in range(0,2):
        current_nuc_seq = [raw_temp[(3*k):(3*k+3)] for k in range(0,54)]
        for j, position in enumerate(position_list):
            current_nuc_seq[position] = codon_dict[best_mutants[i][j]]
        nuc_seqs.append(''.join(current_nuc_seq))

    from Bio import Seq
    from Bio.Seq import Seq

    seq1 = Seq(nuc_seqs[0]).translate()
    seq2 = Seq(nuc_seqs[1]).translate()

    mut1seq = ''.join([seq1[position] for position in position_list])
    mut2seq = ''.join([seq2[position] for position in position_list])
    print('Best sequences:')
    print(mut1seq)
    print(mut2seq)


#Helper function to generate all possible candidate sequences for scoring using
#feature importance values.
def gen_candidate_seqs(best_positions):
    seqs_of_interest = []
    count_array = [0 for 0 in len(best_positions)]
    while count_array[-1] < len(best_positions[-1]):
        current_seq = np.zeros((180))
        for i in range(len(best_positions)):
            current_seq[best_positions[i][count_array[i]] + 20*i] = 1.0
        seqs_of_interest.append(current_seq)
        for i in range(len(best_positions)):
            count_array[i] += 1
            if i == len(options) - 1:
                break
            if count_array[i] > len(best_positions[i]) - 1:
                count_array[i] = 0
            else:
                break
    return seqs_of_interest

    
