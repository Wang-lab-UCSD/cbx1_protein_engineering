import os, sys, numpy as np, Bio, seaborn as sns, matplotlib.pyplot as plt
from Bio.Seq import Seq
from generate_seqdict import parse_sequence_files


#Call this function to subsample the dataset (to evaluate reproducibility etc).
def enrichment_subsampling():
    current_dir = os.getcwd()
    seqdict = parse_sequence_files()
    os.chdir(os.path.join('..', 'processed_data', 'twenty_percent_subsampling'))
    for i in range(0,5):
        subsample_dict = subsample(seqdict, subsample_percentage=0.2)
        generate_enrichment_dataset(subsample_dict, subsample=True, iteration=i)
    os.chdir(current_dir)
    os.chdir(os.path.join('..', 'processed_data', 'ten_percent_subsampling'))
    for i in range(0,5):
        subsample_dict = subsample(seqdict, subsample_percentage=0.1)
        generate_enrichment_dataset(subsample_dict, subsample=True, iteration=i)
    os.chdir(current_dir)


def subsample(seqdict, subsample_percentage):
    subsample_arr = []
    subsample_cats = []
    subsample_dict = dict()
    #To subsample the dataset, we add each sequence to a list x times where x is its
    #frequency. Then we randomly sample the list to create a new list 20% as long as the
    #original. This way, the chances a sequence is chosen are proportional to its
    #frequency, which is just what would happen if we decreased sequencing depth.
    for key in seqdict:
        for i, element in enumerate(seqdict[key]):
            for j in range(0, int(element)):
                subsample_arr.append(key)
                subsample_cats.append(i)
    subsample_arr = np.asarray(subsample_arr)
    print('total size of dataset prior to sampling: %s'%subsample_arr.shape[0])
    subsample_cats = np.asarray(subsample_cats)
    #We use np.random.choice to generate a random subsample selection of the original
    #array.
    chosen_indices = np.random.choice(np.arange(subsample_arr.shape[0]),
                                        int(float(subsample_percentage)*
                                        subsample_arr.shape[0]),replace=False)
    subsample_arr = list(subsample_arr[chosen_indices])
    subsample_cats = list(subsample_cats[chosen_indices])
    #Now we re-combine the chosen subsample back into a sequence dictionary like
    #what was passed to this function.
    for i in range(0, len(subsample_arr)):
        if subsample_arr[i] not in subsample_dict:
            subsample_dict[subsample_arr[i]] = np.zeros((4))
            subsample_dict[subsample_arr[i]][subsample_cats[i]] += 1
        else:
            subsample_dict[subsample_arr[i]][subsample_cats[i]] += 1
    return subsample_dict


def prep_enrichment_data():
    #First, get a dictionary of all acceptable sequences.
    seqdict = parse_sequence_files()
    generate_enrichment_dataset(seqdict, subsample=False)

def generate_enrichment_dataset(seqdict, subsample=False, iteration=0):
    current_dir = os.getcwd()
    aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    #Now calculate enrichment for all sequences and convert this information together with the one-hot encoded
    #sequence into a numpy array.
    seqs_for_enrich = []
    for key in seqdict:
        seq_array = np.zeros((1,182))
        for j, letter in enumerate(key):
            seq_array[0,20*j + aas.index(letter)] = 1.0
        temp_arr = np.sort(seqdict[key][0:3])
        #For enrichment purposes, rather than assigning sequences to categories, we select only sequences that are present both in
        #RH02 and RH03 and calculate an enrichment factor of log(RH03 frequency / RH02 frequency). We will train a model to predict
        #enrichment as a regression objective and determine whether higher predicted enrichment is sufficient to id seqs also present
        #in RH04.
        if seqdict[key][1] > 0 and seqdict[key][2] > 0:
            seq_array[0,180] = np.log10(seqdict[key][2] / seqdict[key][1])
            #If the sequence is found in RH04 at higher frequency, we will record this to evaluate the model but will not provide
            #this information to the model.
            if seqdict[key][3] > seqdict[key][2]:
                seq_array[0,181] = 1
            seqs_for_enrich.append(seq_array)
    final_matrix = np.vstack(seqs_for_enrich)
    print('full dataset')
    print(final_matrix.shape)
    if subsample==False:
        indices = np.random.choice(final_matrix.shape[0], final_matrix.shape[0],
                               replace=False)
        train_cutoff = int(0.8*indices.shape[0])
        os.chdir(os.path.join('..', 'processed_data'))
        np.save('9site_enrich_train.npy', final_matrix[indices[0:train_cutoff],:])
        np.save('9site_enrich_test.npy', final_matrix[indices[train_cutoff:],:])
    else:
        print('subsampled enrichment dataset')
        print(final_matrix.shape)
        print('iteration %s'%iteration)
        np.save('subsample_enrich_%s.npy'%iteration, final_matrix)
    os.chdir(current_dir)




#This function evaluates the relationship between RH02 and RH03 enrichment and RH03 and RH04 enrichment
#(briefly mentioned in the paper) and generates the corresponding graph.
def plot_enrichment_data():
    #First, get a dictionary of all acceptable sequences.
    seqdict = parse_sequence_files()
    #Now, loop through them and if they are present in RH02, RH03 and RH04...
    #check enrichment in RH02 vs RH03 and enrichment in RH03 vs RH04.
    enrichment_rh0203 = []
    enrichment_rh0304 = []
    for key in seqdict:
        if seqdict[key][2] > 0 and seqdict[key][3] > 0:
            enrichment_rh0203.append(np.log((seqdict[key][2]+1) / (seqdict[key][1]+1)))
            enrichment_rh0304.append(np.log((seqdict[key][3]+1) / (seqdict[key][2]+1)))


    #Plot what we get!
    enrichment_rh0203 = np.asarray(enrichment_rh0203)
    enrichment_rh0304 = np.asarray(enrichment_rh0304)
    sns.kdeplot(enrichment_rh0203, enrichment_rh0304)
    plt.title('Enrichment in RH02-RH03 sort vs enrichment in RH03-RH04 sort')
    plt.xlabel('Enrichment in RH02-RH03 sort')
    plt.ylabel('Enrichment in RH03-RH04 sort')
    plt.show()
