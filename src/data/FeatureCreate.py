import numpy as np
import torch
import sys
from . import SimilarityScore as SimilarityScore
from .LoadDisPotential import calculate_edge_potential_matrix_X4
from .LoadDisPotential import calculate_edge_potential_matrix_X3


def seq_id(orderIn1LetterCode_x, orderIn1LetterCode_y):
    I_matrix = np.eye(21)
    xv, yv = np.meshgrid(orderIn1LetterCode_x, orderIn1LetterCode_y,
                         indexing="ij")
    return I_matrix[xv, yv]


def blosum80(orderIn3LetterCode_x, orderIn3LetterCode_y):
    xv, yv = np.meshgrid(
            orderIn3LetterCode_x, orderIn3LetterCode_y, indexing="ij")
    return SimilarityScore.Ori_BLOSUM_80[xv, yv]


def blosum62(orderIn3LetterCode_x, orderIn3LetterCode_y):
    xv, yv = np.meshgrid(
            orderIn3LetterCode_x, orderIn3LetterCode_y, indexing="ij")
    return SimilarityScore.Ori_BLOSUM_62[xv, yv]


def blosum45(orderIn3LetterCode_x, orderIn3LetterCode_y):
    xv, yv = np.meshgrid(
            orderIn3LetterCode_x, orderIn3LetterCode_y, indexing="ij")
    return SimilarityScore.Ori_BLOSUM_45[xv, yv]


def mutationof2pos6(orderIn1LetterCode_x, orderIn1LetterCode_y,
                    tpl_pssm, tgt_pssm):
    x = np.arange(len(orderIn1LetterCode_x))
    xv, yv = np.meshgrid(x, orderIn1LetterCode_y, indexing='ij')
    tpl_pssm = np.insert(tpl_pssm, 20, np.zeros([len(orderIn1LetterCode_x)]),
                         axis=1)
    m = tpl_pssm[xv, yv]

    y = np.arange(len(orderIn1LetterCode_y))
    xv, yv = np.meshgrid(y, orderIn1LetterCode_x, indexing='xy')
    tgt_pssm = np.insert(tgt_pssm, 20, np.zeros([len(orderIn1LetterCode_y)]),
                         axis=1)
    m += tgt_pssm[xv, yv]

    return m


def mutationof2pos6_st(orderIn1LetterCode_x, orderIn1LetterCode_y, tpl_pssm):
    tpl_pssm = np.insert(tpl_pssm, 20, np.zeros([len(orderIn1LetterCode_x)]),
                         axis=1)
    x = np.arange(len(orderIn1LetterCode_x))
    xv, yv = np.meshgrid(x, orderIn1LetterCode_y, indexing='ij')
    m = tpl_pssm[xv, yv]

    return m


def mutationof2pos5(tpl_pssm, tpl_psfm, tgt_pssm, tgt_psfm):
    return np.tensordot(tpl_pssm, tgt_psfm, ([1], [1])) + \
            np.tensordot(tpl_psfm, tgt_pssm, ([1], [1]))


def mutationof2pos5_st(tpl_pssm, tgt_psfm):
    return np.tensordot(tpl_pssm, tgt_psfm, ([1], [1]))


def cc50(orderIn3LetterCode_x, orderIn3LetterCode_y):
    xv, yv = np.meshgrid(
            orderIn3LetterCode_x, orderIn3LetterCode_y, indexing="ij")
    return SimilarityScore.Ori_CC50[xv, yv]


def hdsm(orderIn3LetterCode_x, orderIn3LetterCode_y):
    xv, yv = np.meshgrid(
            orderIn3LetterCode_x, orderIn3LetterCode_y, indexing="ij")
    return SimilarityScore.Ori_HDSM[xv, yv]


def ss3(ss3_x, ss3_y, SS3FeatureMode):
    if SS3FeatureMode == 0:
        return np.array([])
    elif SS3FeatureMode == 6:
        ss3 = np.zeros([len(ss3_x), len(ss3_y), 6])
        tempss3 = np.array(ss3_x)
        seqss3 = np.array(ss3_y)
        xv, yv = np.meshgrid(
                np.arange(len(ss3_x)), np.arange(len(ss3_y)), indexing='ij')
        ss3[xv, yv, :] = np.concatenate((tempss3[xv], seqss3[yv]), -1)
        return ss3
    elif SS3FeatureMode == 9:
        tempss3 = np.array(ss3_x)
        seqss3 = np.array(ss3_y)
        return (tempss3[:, None, :, None]*seqss3[None, :, None, :]).reshape(
                len(ss3_x), len(ss3_y), 9)
    else:
        print("we only support three different SS3FeatureMode: 0, 6, 9")
        sys.exit(-1)


def ss8(ss8_x, ss8_y, SS8FeatureMode):
    if SS8FeatureMode == 0:
        return np.array([])
    elif SS8FeatureMode == 8:
        tempss8 = np.array(ss8_x)
        seqss8 = np.array(ss8_y)
        return np.einsum('ib,jb->ijb', tempss8, seqss8)
    else:
        print("we only support two different SS8FeatureMode: 0, 8")
        sys.exit(-1)


def acc(acc_x, acc_y, ACCFeatureMode):
    if ACCFeatureMode == 0:
        return np.array([])
    if ACCFeatureMode == 6:
        acc = np.zeros([len(acc_x), len(acc_y), 6])
        tempacc = np.zeros([len(acc_x), 3])
        tempacc[np.arange(len(acc_x)), acc_x] = 1
        seqacc = np.array(acc_y)
        xv, yv = np.meshgrid(
                np.arange(len(acc_x)), np.arange(len(acc_y)), indexing='ij')
        acc[xv, yv, :] = np.concatenate((tempacc[xv], seqacc[yv]), -1)
        return acc
    elif ACCFeatureMode == 9:
        tempacc = np.zeros([len(acc_x), 3])
        tempacc[np.arange(len(acc_x)), acc_x] = 1
        seqacc = np.array(acc_y)
        return (tempacc[:, None, :, None] * seqacc[None, :, None, :]).reshape(
                len(acc_x), len(acc_y), 9)
    else:
        print("we only support three different ACCFeatureMode: 0, 6, 9")
        sys.exit(-1)


# return a binary alignment matrix with shape (xLen, yLen),
# alignment is the initial alignment, which can generate by DRNF,
# CNFpred, HHpred or any other TBM method
# An entry of 1 indicates two positions aligned.
# For the input of this function, the alignment should start with 1
def alignmentMatrix(alignment, xLen, yLen, FeatureMode=1):
    if FeatureMode == 1:
        alignment_matrix = torch.zeros(xLen, yLen, 1)
        for pair in alignment:
            x_pos, y_pos, state = pair
            if state == 0:
                alignment_matrix[x_pos, y_pos, 0] = 1
    elif FeatureMode == 3:
        alignment_matrix = torch.zeros(xLen, yLen, 3)
        for pair in alignment:
            x_pos, y_pos, state = pair
            if state < 3:
                alignment_matrix[x_pos, y_pos, state] = 1
    return alignment_matrix


# alignment is the init alignment, which is predicted by DRNF or any other
# threading such as CNFpred or DeepThreader
# For one template residue i, N(i) denote the set of template residues
# with Euclidean distance <=15 with i, For any query residue j, and each
# template edge (i, k), let a(k) be the query residue aligned to k,
# then u can get a distance potential for edge (i, k) using the distance
# between i and k, and the predicted distance potential for j and a(k).
# Summing up all the distance potential for (i, k) over all k in N(i) to
# get the accumulative distance potential for the residue pair (i, j).
def distanceFeature(
        alignment, pair_dis, dis_matrix, disc_method,
        DistFeatureMode, NormFeatureMode):
    alignLen = alignment.size(0)
    xLen = dis_matrix.size(0)
    yLen = pair_dis.size(0)

    if DistFeatureMode == 4 and NormFeatureMode == 4:
        dist_feat = torch.zeros(xLen, yLen, 4)
        norm_dist_feat = torch.zeros(xLen, yLen, 4)
        pot_num = torch.zeros(xLen, yLen, 4)
        near_space, short_space, medium_space, long_space = \
            get_search_space(dis_matrix, disc_method)
        for i in range(alignLen):
            x_pos, y_pos, state = alignment[i]
            if state == 0:
                dist, dist_num = calc_dis_pot(
                    x_pos, y_pos, near_space, pair_dis,
                    dis_matrix, disc_method)
                dist_feat[:, :, 0] += dist
                pot_num[:, :, 0] += dist_num

                dist, dist_num = calc_dis_pot(
                    x_pos, y_pos, short_space, pair_dis,
                    dis_matrix, disc_method)
                dist_feat[:, :, 1] += dist
                pot_num[:, :, 1] += dist_num

                dist, dist_num = calc_dis_pot(
                    x_pos, y_pos, medium_space, pair_dis,
                    dis_matrix, disc_method)
                dist_feat[:, :, 2] += dist
                pot_num[:, :, 2] += dist_num

                dist, dist_num = calc_dis_pot(
                    x_pos, y_pos, long_space, pair_dis,
                    dis_matrix, disc_method)
                dist_feat[:, :, 3] += dist
                pot_num[:, :, 3] += dist_num

    elif DistFeatureMode == 1 and NormFeatureMode == 1:
        dist_feat = torch.zeros(xLen, yLen, 1)
        norm_dist_feat = torch.zeros(xLen, yLen, 1)
        pot_num = torch.zeros(xLen, yLen, 1)
        searchspace = get_search_space(dis_matrix, disc_method, 1)
        for i in range(alignLen):
            x_pos, y_pos, state = alignment[i]
            if state == 0:
                dist, dist_num = calc_dis_pot(
                    x_pos, y_pos, searchspace, pair_dis,
                    dis_matrix, disc_method)
                dist_feat[:, :, 0] += dist
                pot_num[:, :, 0] += dist_num

    # calculate the norm distance feature by dist_feat and pot_num
    mask = torch.gt(pot_num, 0)
    divpot = torch.div(dist_feat, pot_num)
    zero = torch.zeros(divpot.size())
    norm_dist_feat = torch.where(mask, divpot, zero)

    return dist_feat, norm_dist_feat


# we can divide the residues in N(i) into 4 groups:
# near-range (sequence separation in [3, 6) ),
# short-range (seq separation in [6, 12) ),
# medium-range (sequence separation in [12, 24) )
# long-range (seq separation>=24 ).
# we reutrn 2 List, the first list is the lower search space
#                   the first list is the upeer search space
def get_search_space(dis_matrix, disc_method, mode=4):
    xLen = dis_matrix.size(0)
    distance_limit = disc_method[-1]
    if mode == 4:
        near_space = calc_search_space(
            xLen, dis_matrix, distance_limit, 3, 6)
        short_space = calc_search_space(
            xLen, dis_matrix, distance_limit, 6, 12)
        medium_space = calc_search_space(
            xLen, dis_matrix, distance_limit, 12, 24)
        long_space = calc_search_space(
            xLen, dis_matrix, distance_limit, 24, 800)
        return near_space, short_space, medium_space, long_space
    elif mode == 1:
        space = [None] * xLen
        for x_pos in range(xLen):
            upper = max(0, x_pos-5)
            upper_space = torch.where(
                    dis_matrix[x_pos][:upper] < distance_limit)[0]
            lower = min(xLen, x_pos+6)
            lower_space = torch.where(
                    dis_matrix[x_pos][lower:] < distance_limit)[0]
            lower_space += int(lower)
            space[x_pos] = [lower_space, upper_space]
        return space


def calc_dis_pot(x_pos, y_pos, search_space,
                 pair_dis, dis_matrix, disc_method):
    T_space_lower, T_space_upper = search_space[x_pos]
    xLen = dis_matrix.size(0)
    yLen = pair_dis.size(0)
    lower = min(y_pos+6, yLen)
    Q_space = torch.LongTensor(list(range(lower, yLen)))
    dist = torch.zeros(xLen, yLen)
    dist_num = torch.zeros(xLen, yLen)
    if len(Q_space) != 0:
        T_space = T_space_lower.expand(
            len(Q_space), len(T_space_lower)).contiguous()
        Q_space = Q_space.unsqueeze(-1).contiguous().expand(
            len(Q_space), len(T_space_lower)).contiguous().view(-1)
        pot = calculate_edge_potential_matrix_X4(
            x_pos, T_space, y_pos, Q_space, pair_dis, dis_matrix, disc_method)
        dist[T_space.view(-1), Q_space] += pot.view(-1)
        dist_num[T_space.view(-1), Q_space] += 1

    upper = max(y_pos-5, 0)
    Q_space = torch.LongTensor(list(range(0, upper)))
    if len(Q_space) != 0:
        T_space = T_space_upper.expand(
            len(Q_space), len(T_space_upper)).contiguous()
        Q_space = Q_space.unsqueeze(-1).contiguous().expand(
            len(Q_space), len(T_space_upper)).contiguous().view(-1)
        pot = calculate_edge_potential_matrix_X3(
            T_space, x_pos, Q_space, y_pos, pair_dis, dis_matrix, disc_method)
        dist[T_space.view(-1), Q_space] += pot.view(-1)
        dist_num[T_space.view(-1), Q_space] += 1
    return dist, dist_num


def calc_search_space(xLen, dis_matrix, distance_limit, i1, i2):
    assert i2-i1 >= 3, "the length of the interval" \
        " has to be greater or equal to 3"
    space = [None] * xLen
    for x_pos in range(xLen):
        # near-range
        lower_space = torch.where(
            dis_matrix[x_pos][
                max(0, x_pos-i2+1):max(0, x_pos-i1+1)] < distance_limit)[0]
        lower_space += int(max(0, x_pos-i2+1))
        upper_space = torch.where(
            dis_matrix[x_pos][
                min(xLen, x_pos+i1):min(xLen, x_pos+i2)] < distance_limit)[0]
        upper_space += int(min(xLen, x_pos+i1))
        space[x_pos] = [lower_space, upper_space]
    return space
