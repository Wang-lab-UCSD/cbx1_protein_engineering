import os, sys, numpy as np, Bio, seaborn as sns, matplotlib.pyplot as plt
from Bio.Seq import Seq
from generate_seqdict import parse_sequence_files


def calculate_enrichment():
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
