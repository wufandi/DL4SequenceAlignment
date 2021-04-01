import torch
import os
import numpy as np
from scipy.spatial import distance_matrix
from .interface import compute_pairwise_score


# calculate the Cb-Cb distance matrix for an template
def Compute_CbCb_distance_matrix(tpl):
    invalid = [-999, -999, -999]
    InvalidDistance = -1
    eps = 0.001

    # for GLY we use Ca
    valid = ~np.char.equal(np.array(list(tpl['sequence'])), "G")
    Cb = np.where(np.stack((valid, valid, valid), -1), tpl['Cb'], tpl['Ca'])

    # calculate CbCb distance
    CbCbDist = distance_matrix(Cb, Cb).astype(np.float16)
    CbValid = [all(abs(cb - invalid) > 0.1) for cb in Cb]
    CbCbValid = np.outer(CbValid,  CbValid)
    np.putmask(CbCbDist, ~CbCbValid, InvalidDistance)
    np.fill_diagonal(CbCbDist, eps)

    return CbCbDist


# calculate the Cb-Cb distance for each position in templates
def DistOf2AAs(tpl, i, j):
    if tpl["missing"][i] == 0 and tpl["missing"][j] == 0:
        if tpl["sequence"][i] == "G":
            atom_i = tpl["Ca"][i]
        else:
            atom_i = tpl["Cb"][i]
        if tpl["sequence"][j] == "G":
            atom_j = tpl["Ca"][j]
        else:
            atom_j = tpl["Cb"][j]
        return np.sqrt((atom_i[0]-atom_j[0])*(atom_i[0]-atom_j[0]) +
                       (atom_i[1]-atom_j[1])*(atom_i[1]-atom_j[1]) +
                       (atom_i[2]-atom_j[2])*(atom_i[2]-atom_j[2]))
    else:
        return -1


# detact multi HIS, scan from left to right
# output is a 0-1 tensor
def DetectMultiHIS(sequence):
    HISflag = torch.zeros((len(sequence),), dtype=torch.uint8)
    pos1 = sequence.find('HHH')
    if pos1 >= 0 and pos1 < 10:
        i = pos1 + 3
        while i < len(sequence) and sequence[i] == 'H':
            i = i + 1
        HISflag[:i] = 1
    pos2 = sequence.rfind('HHH')
    if pos2 != -1 and pos2 > len(sequence) - 10:
        i = pos2 - 1
        while i >= 0 and sequence[i] == 'H':
            i = i - 1
        HISflag[i+1:] = 1
    return HISflag.bool()


# detact X position in templates
def DetectX(sequence):
    Xflag = torch.zeros((len(sequence),), dtype=torch.uint8)
    for i in range(len(sequence)):
        if sequence[i] == "X":
            Xflag[i] = 1
    return Xflag.bool()


# this function get a alignment tensor and generate alignment file
# one alignment file example is as follows
# >XXXXA
# AAAAAAAAAAAAAACCCCCCCDDDDDDDDDDDDGGGGGGGGGGGHH--HHHHHHHHH
# >query
# AA--AAAAAAAAACCCCC-CCDD-DDDD-DDDGGGGGGGGGGGGHHHHHHHH-----
def alignment_output(tplname, tgtname, tpl, tgt, alignment):
    output = ">" + tplname + "\n"
    AlignLen, _ = alignment.size()
    tplseq = ""
    tgtseq = ""
    indexX = 0
    indexY = 0
    for i in range(AlignLen):
        # match
        if alignment[i][2] == 0:
            tplseq = tplseq + tpl[indexX]
            tgtseq = tgtseq + tgt[indexY]
            indexX += 1
            indexY += 1
        # insert X
        elif alignment[i][2] == 1:
            tplseq = tplseq + tpl[indexX]
            tgtseq = tgtseq + "-"
            indexX += 1
        # insert Y
        elif alignment[i][2] == 2:
            tplseq = tplseq + "-"
            tgtseq = tgtseq + tgt[indexY]
            indexY += 1
        # headgap and tailgap
        elif alignment[i][2] == 3 or alignment[i][2] == 4:
            if alignment[i][1] == -1:
                tplseq = tplseq + tpl[indexX]
                tgtseq = tgtseq + "-"
                indexX += 1
            if alignment[i][0] == -1:
                tplseq = tplseq + "-"
                tgtseq = tgtseq + tgt[indexY]
                indexY += 1
    output += tplseq + "\n"
    output += ">" + tgtname + "\n"
    # alignfile.write("sequence:::::::::\n")
    output += tgtseq + "\n"
    return output


# get the number of different state(Match, InsertX, InsertX) from an alignment
def getStateNum(alignment):
    AlignLen = alignment.size(0)
    Match = 0
    InsertX = 0
    InsertY = 0
    for i in range(AlignLen):
        x_pos, y_pos, state = alignment[i]
        if state == 0:
            Match += 1
        elif state == 1:
            InsertX += 1
        elif state == 2:
            InsertY += 1
    return Match, InsertX, InsertY


# the the identical amino acid from an alignment
def getIdenticals(alignment, tplseq, tgtseq):
    AlignLen = alignment.size(0)
    identicals = 0
    for i in range(AlignLen):
        x_pos, y_pos, state = alignment[i]
        if state == 0:
            if tplseq[x_pos] == tgtseq[y_pos]:
                identicals += 1
    return identicals


def getGapTransition(alignment):
    AlignLen = alignment.size(0)
    transition = 0
    for i in range(AlignLen-1):
        if (alignment[i][2] == 1 and alignment[i+1][2] != 1) or \
                (alignment[i][2] != 1 and alignment[i+1][2] == 1):
            transition += 1
    return transition


# get the Alignment Start position
# if not align at all, return 0, 0
def getAlignmentStart(alignment):
    alignLen = alignment.size(0)
    query_start = 0
    subject_start = 0
    for i in range(alignLen):
        if alignment[i, 2] == 0:
            query_start = alignment[i, 1].item() + 1
            subject_start = alignment[i, 0].item() + 1
            break
    return query_start, subject_start


# get the alignment End position
# if not align at all, return 0, 0
def getAlignmentEnd(alignment):
    alignLen = alignment.size(0)
    query_end = 0
    subject_end = 0
    for i in range(alignLen-1, -1, -1):
        if alignment[i, 2] == 0:
            query_end = alignment[i, 1].item() + 1
            subject_end = alignment[i, 0].item() + 1
            break
    return query_end, subject_end


# compute the alignment singleton_score
# we deal with missing and bad, so it may exist InsertY->InsertX on alignment
def Alignment_Singleton_Score(alignments, observations, transitions):
    batchSize, xLen, yLen, numStates = observations.size()
    singleton_score = torch.zeros(batchSize)

    for ba in range(batchSize):
        AlignLen = alignments[ba].size(0)
        obsScore = torch.zeros(AlignLen)
        for i in range(AlignLen):
            x_pos, y_pos, state = alignments[ba][i]
            if state < 3:
                obsScore[i] = observations[ba, x_pos, y_pos, state]
        prevState = alignments[ba][:-1, -1]
        currState = alignments[ba][1:, -1]
        trans = transitions[ba]
        transScore = trans[prevState, currState]
        singleton_score[ba] = torch.sum(obsScore) + torch.sum(transScore)
    return singleton_score


# compute the alignment pairwise_score
def Alignment_Pairwise_Score(alignments, pair_dis, dis_matrix,
                             disc_method, edge_type="dist", Norm_Weight=1):
    batchSize = len(alignments)
    pairwise_score = torch.zeros(batchSize)
    norm_score = torch.zeros(batchSize)
    pair_dis_numpy = pair_dis.numpy().astype(np.float)
    disc_method_numpy = np.array(disc_method).astype(np.float)
    if edge_type == "dist":
        use_dist_pot = True
    else:
        use_dist_pot = False
    for ba in range(batchSize):
        alignment = alignments[ba].numpy().astype(np.int)
        pairwise_score[ba], norm_score[ba] = compute_pairwise_score(
                alignment, pair_dis_numpy,
                dis_matrix.numpy().astype(float),
                disc_method_numpy, Norm_Weight, use_dist_pot)
    return pairwise_score, norm_score


# sort the threading output by total score
def sortoutput(tgtname, listfile, k, sort_col=12):
    output = []
    outputfile = open(listfile, 'r')
    workdir = os.path.dirname(listfile)
    sortoutputfile = os.path.join(workdir, tgtname + '_list.SortedScore')
    topKfile = os.path.join(workdir, tgtname + '_list.SortedScore_topK')
    for line in outputfile.readlines():
        score = float(line.split()[sort_col-1])
        output.append([line, score])
    outputfile.close()
    output.sort(key=lambda item: item[1], reverse=True)
    with open(sortoutputfile, 'w') as f:
        for i in range(len(output)):
            f.write(output[i][0])
    with open(topKfile, 'w') as f:
        for i in range(min(k, len(output))):
            f.write(output[i][0])
    return
