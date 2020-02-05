import Bio, gzip, os, numpy as np
from Bio.Seq import Seq
from Bio import SeqIO

def check_raw_reads(target_dir):
    #This is what we expect the ends of the paired-end reads to look like.
    #If they DON'T have these, we toss them for quality control.
    prime5 = 'CGATGACGATAAGGTACCAGGATCCAGT'
    prime3 = 'TCTAGAGGGCCCTTCGAAGGTAAGC'
    os.chdir(target_dir)
    filenames = [x for x in os.listdir() if x.endswith('.fastq')]
    #Check to make sure we have both the left end paired end read and the right
    #end paired end read file, and order them in the correct order.
    if 'R1' in names[0]:
        finalnamelist = filenames
    elif 'R2' in names[0]:
        finalnamelist = [filenames[1], filenames[0]]
    #Collect all the reads...
    leftreads, rightreads = [], []
    with open(finalnamelist[0],'r') as leftreadfile:
        with open(finalnamelist[1], 'r') as rightreadfile:
            for record in SeqIO.parse(leftreadfile, 'fastq'):
                leftreads.append(record)
            for record in SeqIO.parse(rightreadfile, 'fastq'):
                rightreads.append(record)

    print('length of list from leftreadfile: %s'%len(leftreads))
    print('length of list from rightreadfile: %s'%len(rightreads))
    finalseqs = dict()
    #Track sequences that are tossed for various reasons to give
    #a printout of what happened during processing.
    totseqs, weirdlength, lowqual, nonmerged, wrongstart, badprime = 0, 0, 0, 0, 0, 0
    lengthdistro = np.zeros((110))
    for i in range(0, len(leftreads)):
        leftphred = np.asarray(leftreads[i].letter_annotations['phred_quality'])
        rightphred = np.asarray(rightreads[i].letter_annotations['phred_quality'])
        leftseq = leftreads[i]
        rightseq = rightreads[i]
        #Check to make sure phred read quality is acceptable.
        if np.min(leftphred) > 10 and np.min(rightphred) > 10:
            leftraw = leftreads[i].seq
            rightraw = rightreads[i].seq.reverse_complement()
            keepseq = False
            if leftraw[0:28] == prime5 and rightraw[-25:] == prime3:
                leftseq = str(leftraw[28:].translate())
                rightseq = str(rightraw[0:-25].translate())
                #Find a matching overlap if there is one. If not,
                #keepseq is not set to true, and we will discard
                #this pair of reads, because they do not match.
                for j in range(45,20,-1):
                    if leftseq[-j:] == rightseq[0:j]:
                        keepseq = True
                        break
                if keepseq == True:
                    merge_point = len(leftseq) - j
                    merged_seq = ''.join([leftseq[0:merge_point],
                                     rightseq])
                    if len(merged_seq) == 54 and '*' not in merged_seq:
                        totseqs += 1
                        if merged_seq not in finalseqs:
                            finalseqs[merged_seq] = 1
                            else:
                                finalseqs[merged_seq] += 1
                            else:
                                weirdlength += 1
            else:
                badprime += 1
            if keepseq == False:
                nonmerged += 1
        else:
                lowqual += 1
        if i % 10000 == 0:
                print(i)
    #Tell the user what we found, then save sequences to file.
    print('%s weirdlength'%weirdlength)
    print('%s badprime'%badprime)
    print('%s nonmerged'%nonmerged)
    print('%s lowqual'%lowqual)
    print('%s wrong start or end'%wrongstart)
    print('%s total sequences found'%totseqs)
    print('%s unique sequences found'%len(finalseqs))
    with open('%s_sequences.txt'%target_dir, 'w+') as handle:
        for i, key in enumerate(finalseqs):
            handle.write(key + '\t' + str(finalseqs[key]) + '\n')


if __name__ == '__main__':
    main()
