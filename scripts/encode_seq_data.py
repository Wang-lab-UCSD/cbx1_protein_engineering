import os, sys, numpy as np, Bio, pickle
from Bio.Seq import Seq
from generate_seqdict import parse_sequence_files
from known_tight_binders import generate_known_tight_binders

#The order of AAs in this list determines the one-hot encoding.
aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

#This is the WT sequence.
aa_template = 'EYVVEKVLDRRVVKGKVEYLLKWKGFSDEDNTWEPEENLDCPDLIAEFLQSQKT'
#These are the amino acids we expect not to vary from the WT.
invariant_aas = ''.join([aa_template[i] for i in range(0, 54) if
                                  i not in set([0,2,5,29,36,37,
                                            39,40,42])])

def nine_site_sequence_encoding():
    seqdict = parse_sequence_files()
    os.chdir(os.path.join('..', 'processed_data'))
    #Generate and save the known tight binders we will need to score later.
    known_tight_binders = generate_known_tight_binders()
    known_tight_data = np.zeros((9,186))
    for i, key in enumerate(known_tight_binders):
        for j, letter in enumerate(key):
            known_tight_data[i,20*j + aas.index(letter)] = 1.0
    np.save('known_high_binders_9site.npy', known_tight_data)
    #Process the sequence dict and save to a file.
    process_seqdict(seqdict)

#Call this function to subsample the dataset (to evaluate reproducibility etc).
def subsample_sequence_encoding():
    current_dir = os.getcwd()
    seqdict = parse_sequence_files()
    os.chdir(os.path.join('..', 'processed_data', 'twenty_percent_subsampling'))
    for i in range(0,5):
        subsample_dict = subsample(seqdict, subsample_percentage=0.2)
        process_seqdict(subsample_dict, subsample=True, iteration=i)
    os.chdir(current_dir)
    os.chdir(os.path.join('..', 'processed_data', 'ten_percent_subsampling'))
    for i in range(0,5):
        subsample_dict = subsample(seqdict, subsample_percentage=0.1)
        process_seqdict(subsample_dict, subsample=True, iteration=i)
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

def process_seqdict(seqdict, subsample=False, iteration=0):
    #We'll process the data twice -- once to generate the data matrix used by the nominal
    #classification & random forest models, the other to generate the data matrix used by
    #the ordinal regression model. The difference is how category is defined.
    seqs_for_ord_reg, seqs_for_nominal_class = [], []
    for key in seqdict:
        seq_for_ord = np.zeros((1,186))
        seq_for_nom = np.zeros((1,183))
        for j, letter in enumerate(key):
            seq_for_ord[0,20*j + aas.index(letter)] = 1.0
        seq_for_nom[0,0:180] = seq_for_ord[0,0:180]
        temp_arr = np.sort(seqdict[key][0:3])
        cat = -1
        #If a sequence is present in only one category at frequency > 1,
        #or if the sequence is present at least twice as frequently in
        #one category as in the next, we can assign it to that category.
        #Otherwise we just don't have enough info to make a confident assignment
        #so we drop it. This loses some data but also cuts out a lot of noise.
        if temp_arr[-2] == 0:
            if temp_arr[-1] > 1:
                cat = np.argmax(seqdict[key][0:3])
        elif (temp_arr[-1] > 2*temp_arr[-2]) and temp_arr[-1] > 1:
            cat = np.argmax(seqdict[key][0:3])
        #If sequence was kept, process accordingly.
        if cat != -1:
            #For nominal sequences, store the non-one-hot encoded category
            #assignment under column 180; for ordinal, under column 184.
            seq_for_ord[0,184] = cat
            seq_for_nom[0,180] = cat
            #If sequences are found in RH04, we add this information although we will
            #not provide this to the model.
            if seqdict[key][-1] > (temp_arr[-1]):
                seq_for_ord[0,183] = 3
                seq_for_nom[0,181] = 3
            #Encode the category for ordinal regression as described in
            #the paper.
            seq_for_ord[0,180:(180+cat)] = 1.0
            #We weight sequences based on how confidently they can be assigned to a
            #category (ratio of frequency in a given category to closest category).
            seq_for_ord[0,185] = (temp_arr[-1]+1) / (temp_arr[-2]+1)
            seq_for_nom[0,182] = (temp_arr[-1]+1) / (temp_arr[-2]+1)
            seqs_for_ord_reg.append(seq_for_ord)
            seqs_for_nominal_class.append(seq_for_nom)
    final_matrix = np.vstack(seqs_for_ord_reg)
    final_nom_matrix = np.vstack(seqs_for_nominal_class)
    #Split the resulting nominal and ordinal datasets up into train
    #and test sets by creating a random index and using the first 80%
    #for train, the last 20% for test.
    if subsample == False:
        print('full dataset')
        print(final_matrix.shape)
        indices = np.random.choice(final_matrix.shape[0], final_matrix.shape[0],
                               replace=False)
        train_cutoff = int(0.8*indices.shape[0])
        np.save('9site_train.npy', final_matrix[indices[0:train_cutoff],:])
        np.save('9site_test.npy', final_matrix[indices[train_cutoff:],:])
        np.save('9site_nominal_class_train.npy',
            final_nom_matrix[indices[0:train_cutoff],:])
        np.save('9site_nominal_class_test.npy',
            final_nom_matrix[indices[train_cutoff:],:])
        print('rh04: %s'%np.argwhere(final_matrix[:,183]==3).shape[0])
    else:
        print('subsampled dataset')
        print(final_matrix.shape)
        print('iteration %s'%iteration)
        np.save('subsample_ord_%s.npy'%iteration, final_matrix)
        np.save('subsample_nom_%s.npy'%iteration, final_nom_matrix)


if __name__ == '__main__':
    nine_site_sequence_encoding()
