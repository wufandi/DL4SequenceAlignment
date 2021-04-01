import torch
import sys


# this function loads some file containing some alignments.
# Each alignment has two proteins. The first one is assumed to be a template
# and the 2nd one is a query sequence.
# alignfiles: a list of text file contains a set of alignments
# pairnames: the pairnames to check the alignment file
# since each alignment has 4 lines.
# one alignment example is as follows
# >XXXXA
# AAAAAAAAAAAAAACCCCCCCDDDDDDDDDDDDGGGGGGGGGGGHH--HHHHHHHHH
# >query
# AA--AAAAAAAAACCCCC-CCDD-DDDD-DDDGGGGGGGGGGGGHHHHHHHH-----
# this function return a tensor with shape (batchSize, AlignLen, 3)
# The last dimension consists of two residue indice and one state
# the numstate = 5  match(0), insertX(1) and insertY(2), headGap(3)
# and tailGap(4), and padding state(5)
# residue index starts from 1 and ends at sequence length.
# a residue index of -1 is used to indicate head and tail gaps
def ReadAlignments(alignfiles, pairnames):
    alignments = LoadAlignFile(alignfiles)
    # init the output with shape (batchSize, AlignLen, 3)
    batchSize = len(alignments)
    assert len(pairnames) == batchSize, \
        "number of pairnames %d should equal to %d " % (
                len(pairnames), batchSize)

    allLen = max([len(alignments[i][0]) for i in range(batchSize)])
    output = torch.zeros((batchSize, allLen, 3))
    output[:, :] = torch.Tensor([0, 0, 5])
    for batch in range(batchSize):
        # check the sequence in alignments
        assert len(alignments[batch]) > 1
        assert len(alignments[batch][0]) == len(alignments[batch][1])
        seqX_name = alignments[batch][2]
        seqY_name = alignments[batch][3]
        # check the name in alignments
        assert seqX_name == pairnames[batch][0], \
            "alignments seqY: %s should equal to seqY_name: %s" % (
                    seqX_name, pairnames[batch][0])
        assert seqY_name == pairnames[batch][1], \
            "alignments seqX: %s should equal to seqX_name: %s" % (
                    seqY_name, pairnames[batch][1])
        # sequence of tpl
        seqX = alignments[batch][0]
        # sequence of tgt
        seqY = alignments[batch][1]
        firstMatch = 0
        alignLen = len(alignments[batch][0])
        alignNum = 0
        indexX = 0
        indexY = 0
        while (alignNum != alignLen):
            # consider the state of head gaps
            if seqX[alignNum] == '-' and seqY[alignNum] != '-':
                if firstMatch == 0:
                    indexY += 1
                    output[batch][alignNum] = torch.Tensor([-1, indexY, 3])
                else:
                    indexY += 1
                    output[batch][alignNum] = torch.Tensor([indexX, indexY, 2])
            elif seqX[alignNum] != '-' and seqY[alignNum] == '-':
                if firstMatch == 0:
                    indexX += 1
                    output[batch][alignNum] = torch.Tensor([indexX, -1, 3])
                else:
                    indexX += 1
                    output[batch][alignNum] = torch.Tensor([indexX, indexY, 1])
            elif seqX[alignNum] != '-' and seqY[alignNum] != '-':
                firstMatch = 1
                indexX += 1
                indexY += 1
                output[batch][alignNum] = torch.Tensor([indexX, indexY, 0])
            else:
                print(indexX, seqX[indexX], indexY, seqY[indexY])
                sys.exit("the alignment data is wrong in position "
                         + str(alignNum))
            alignNum += 1
        lastMatch = 0
        # consider the state of tail gaps
        while (lastMatch == 0):
            alignNum -= 1
            if seqX[alignNum] == '-' and seqY[alignNum] != '-':
                output[batch][alignNum] = torch.Tensor([-1, indexY, 4])
                indexY -= 1
            elif seqX[alignNum] != '-' and seqY[alignNum] == '-':
                output[batch][alignNum] = torch.Tensor([indexX, -1, 4])
                indexX -= 1
            else:
                lastMatch = 1
    return output.long()


# the output of ReadAlignments return an alignment start with 1
# it's used for training, but we also need an alignment start with 0
def refactor_alignment(alignments):
    batchSize, AlignLen, numStates = alignments.size()
    if batchSize != 1:
        print("this function only used for 1 alignment")
        sys.exit(-1)
    for i in range(AlignLen):
        x_pos, y_pos, state = alignments[0][i]
        if state < 3:
            if x_pos != -1:
                alignments[0][i][0] -= 1
            if y_pos != -1:
                alignments[0][i][1] -= 1
    return alignments


def refactor_Single_alignment(alignments):
    AlignLen, numStates = alignments.size()
    realLen = 0
    for i in range(AlignLen):
        x_pos, y_pos, state = alignments[i]
        if state < 3:
            if x_pos != -1:
                alignments[i][0] -= 1
            if y_pos != -1:
                alignments[i][1] -= 1
        if state < 5:
            realLen += 1
    return alignments[:realLen]


# this function loads some file containing some alignments.
# return a list of alignment information
def LoadAlignFile(alignment_files):
    alignments = []
    for ba in range(len(alignment_files)):
        fin = open(alignment_files[ba], 'r')
        content = [line.strip() for line in list(fin)]
        fin.close()

        # remove empty lines
        alignment_result = [c for c in content if c]
        if len(alignment_result) % 4 != 0:
            print('the number of lines in the alignment file is'
                  'incorrect: ', alignment_files[ba])
            exit(-1)

        for i in range(0, len(alignment_result), 4):
            tpl_name = alignment_result[i][1:]
            tpl_seq = alignment_result[i+1]
            tgt_name = alignment_result[i+2][1:]
            tgt_seq = alignment_result[i+3]

        alignments.append((tpl_seq, tgt_seq, tpl_name, tgt_name))

    return alignments
