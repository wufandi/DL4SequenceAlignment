import torch
import numpy as np
import sys
from . import interface as cpp
from . import Crf4SeqAlign as CrfModel
sys.path.append("../../")
from src.data.LoadDisPotential import calculate_edge_potential_matrix_X3, \
    calculate_edge_potential_matrix_X4          # noqa: E402


# alignments is a list with batchSize alignment
# dis_matrix is a list with batchSize dis_matrix
# deal with sub-problems1:
#   y = arg max{sigma y_{kl}^v C_{kl}^v}
#   whle C_{kl}^v = 1/L sigma theta_{ijkl}^{uv} z_{ij}^u - lambda_{kl}^{v}
#                   - rho / 2 * (1-2 * z_{kl}^v)
# we use search_space to speed up pairwise_score calculation
# search_space is tensor with shape (batchSize, yLen, Limit)
def update_potential_y(alignments, singleton_score, penalty, Lambda,
                       search_space, pair_dis, dis_matrix,
                       disc_method, Rho=0.01, edge_type="dist", Edge_Norm=1):
    batchSize, xLen, yLen, _ = singleton_score.size()

    potential_Y = torch.zeros((batchSize, xLen, yLen, 3))
    obs_Z = alignment2obs(alignments, xLen, yLen).float()
    pot_num = torch.zeros(batchSize, xLen, yLen)

    for ba in range(batchSize):
        AlignLen = alignments[ba].size(0)
        for i in range(AlignLen):
            x_pos, y_pos, state = alignments[ba][i]
            # theta_{ijkl}^uv is equal to 0 if either u or v is not match
            if state == 0:
                T_space_lower, T_space_upper = search_space[x_pos]
                # first, we consider position equal or larger then x_pos
                # T_space has shape (Q_space_len, Limit)
                # to matrixize the calculation, we should expand the Q_space
                # we only consider the position equal or larger
                # then y_pos in sequence, so we use ModifySpace to
                # get the True search_space
                lower = min(y_pos+6, yLen)
                Q_space = torch.LongTensor(list(range(lower, yLen)))
                if len(Q_space) != 0:
                    T_space = T_space_lower.expand(
                            len(Q_space), len(T_space_lower)).contiguous()
                    Q_space = Q_space.unsqueeze(-1).contiguous().expand(
                            len(Q_space), len(T_space_lower)
                            ).contiguous().view(-1)
                    pot = calculate_edge_potential_matrix_X4(
                            x_pos, T_space, y_pos, Q_space,
                            pair_dis, dis_matrix, disc_method,
                            edge_type, Edge_Norm)
                    potential_Y[ba, T_space.view(-1), Q_space, 0] +=\
                        pot.view(-1)
                    pot_num[ba, T_space.view(-1), Q_space] += 1

                # second, we consider position lower then x_pos
                # we only consider the position lower then y_pos in sequence
                # so we use ModifySpace to get the True search_space
                upper = max(y_pos-5, 0)
                Q_space = torch.LongTensor(list(range(0, upper)))
                if len(Q_space) != 0:
                    T_space = T_space_upper.expand(
                            len(Q_space), len(T_space_upper)).contiguous()
                    Q_space = Q_space.unsqueeze(-1).contiguous().expand(
                            len(Q_space), len(T_space_upper)
                            ).contiguous().view(-1)
                    pot = calculate_edge_potential_matrix_X3(
                            T_space, x_pos, Q_space, y_pos,
                            pair_dis, dis_matrix, disc_method,
                            edge_type, Edge_Norm)
                    potential_Y[ba, T_space.view(-1), Q_space, 0] += \
                        pot.view(-1)
                    pot_num[ba, T_space.view(-1), Q_space] += 1

        # the pairwise_score should div length of alignment,
        # we use search_space rather than compute all position,
        # so the true length should be pot_num
        if edge_type == "epad":
            mask = torch.gt(pot_num[ba], 0)
            divpot = torch.div(potential_Y[ba, :, :, 0], pot_num[ba])
            zero = torch.zeros(divpot.size())
            potential_Y[ba, :, :, 0] = torch.where(mask, divpot, zero)

    potential_Y.sub_(Lambda + 0.5*Rho)
    potential_Y.add_(Rho * obs_Z)
    potential_Y.add_(singleton_score)
    alignmentY = Compute_Viterbi(potential_Y, penalty)

    return alignmentY


# alignments is a list with batchSize alignment
# dis_matrix is a list with batchSize dis_matrix
# deal with sub-problems2:
#   z = arg max{sigma z_{ij}^u D_{ij}^u}
#   whle D_{ij}^u =theta_{ij}^u +  1/L sigma theta_{ijkl}^{uv} y_{kl}^v
#                  - lambda_{ij}^{u} - rho / 2 * (1 - 2 * y_{ij}^u)
# we use search_space to speed up pairwise_score calculation
def update_potential_z(alignments, singleton_score, penalty,
                       Lambda, search_space, pair_dis, dis_matrix,
                       disc_method, Rho=0.1, edge_type="dist", Edge_Norm=1):
    batchSize, xLen, yLen, _ = singleton_score.size()

    potential_Z = torch.zeros((batchSize, xLen, yLen, 3))
    obs_Y = alignment2obs(alignments, xLen, yLen).float()
    pot_num = torch.zeros(batchSize, xLen, yLen)

    for ba in range(batchSize):
        AlignLen = alignments[ba].size(0)
        for i in range(AlignLen):
            x_pos, y_pos, state = alignments[ba][i]
            # theta_{ijkl}^uv is equal to 0 if either u or v is not match
            if state == 0:
                T_space_lower, T_space_upper = search_space[x_pos]
                # first, we consider position lower then x_pos
                # Q_space has shape (T_space, Limit)
                # to matrixize the calculation, we should expand the T_space
                # we only consider the position lower
                # then y_pos in sequence, so we use ModifySpace to
                # get the True search_space
                upper = max(y_pos-5, 0)
                Q_space = torch.LongTensor(list(range(0, upper)))
                if len(Q_space) != 0:
                    T_space = T_space_upper.expand(
                            len(Q_space), len(T_space_upper)).contiguous()
                    Q_space = Q_space.unsqueeze(-1).contiguous().expand(
                            len(Q_space), len(T_space_upper)
                            ).contiguous().view(-1)
                    pot = calculate_edge_potential_matrix_X3(
                            T_space, x_pos, Q_space, y_pos,
                            pair_dis, dis_matrix, disc_method,
                            edge_type, Edge_Norm)
                    potential_Z[ba, T_space.view(-1), Q_space, 0] += \
                        pot.view(-1)
                    pot_num[ba, T_space.view(-1), Q_space] += 1

                # second, we consider position equal or larger then x_pos
                # we only consider the position equal or larger
                # then y_pos in sequence
                # so we use ModifySpace to get the True search_space
                lower = min(y_pos+6, yLen)
                Q_space = torch.LongTensor(list(range(lower, yLen)))
                if len(Q_space) != 0:
                    T_space = T_space_lower.expand(
                            len(Q_space), len(T_space_lower)).contiguous()
                    Q_space = Q_space.unsqueeze(-1).contiguous().expand(
                            len(Q_space), len(T_space_lower)
                            ).contiguous().view(-1)
                    pot = calculate_edge_potential_matrix_X4(
                            x_pos, T_space, y_pos, Q_space,
                            pair_dis, dis_matrix, disc_method,
                            edge_type, Edge_Norm)
                    potential_Z[ba, T_space.view(-1), Q_space, 0] += \
                        pot.view(-1)
                    pot_num[ba, T_space.view(-1), Q_space] += 1
        # the pairwise_score should div length of alignment,
        # we use search_space rather than compute all position,
        # so the true length should be pot_num
        if edge_type == "epad":
            mask = torch.gt(pot_num[ba], 0)
            divpot = torch.div(potential_Z[ba, :, :, 0], pot_num[ba])
            zero = torch.zeros(divpot.size())
            potential_Z[ba, :, :, 0] = torch.where(mask, divpot, zero)

    potential_Z.add_(Lambda - Rho * 0.5)
    potential_Z.add_(singleton_score)
    potential_Z.add_(Rho * obs_Y)
    alignmentZ = Compute_Viterbi(potential_Z, penalty)

    return alignmentZ


# the special viterbi with mask for admm
# search_space has shape (batchSize, yLen, Limit)
def Compute_Viterbi(potential, penalty):
    batchSize, xLen, yLen, numStates = potential.size()
    maskX = [xLen] * batchSize
    pot = potential.detach().numpy()
    tra = penalty.detach().numpy()
    mas = np.array(maskX).astype(np.int)

    maxScores, argmaxPos, tracebacks = cpp.viterbi(pot, tra, mas)
    maxScores = torch.from_numpy(maxScores)
    argmaxPos = torch.from_numpy(argmaxPos)
    tracebacks = torch.from_numpy(tracebacks)
    alignments = []

    for score_max, pos_max, traceback, realLenX in zip(maxScores, argmaxPos,
                                                       tracebacks, maskX):
        if score_max <= 0.0:
            alignments.append(CrfModel.ExpandAlignment([], realLenX, yLen))
            continue

        curr_pos = (pos_max[0], pos_max[1], 0)
        pos_list = [curr_pos]
        while traceback[curr_pos] < 3:
            curr_state = curr_pos[2]
            if curr_state == 0:
                x = curr_pos[0] - 1
                y = curr_pos[1] - 1
            elif curr_state == 1:
                x = curr_pos[0] - 1
                y = curr_pos[1]
            else:
                x = curr_pos[0]
                y = curr_pos[1] - 1
            curr_pos = (x, y, traceback[curr_pos])
            pos_list.insert(0, curr_pos)

        alignment = CrfModel.ExpandAlignment(pos_list, realLenX, yLen)
        alignments.append(alignment)

    return alignments


# return a 0-1 obs tensor, if the residues is exist,
# the x_pos, y_pos, state is 1
# otherwise, it's 0
# Hint: the alignments is not a tensor, but a list of tensor
def alignment2obs(alignments, xLen, yLen, numStates=3):
    batchSize = len(alignments)
    obs_Alignments = torch.zeros((batchSize, xLen, yLen, numStates),
                                 dtype=torch.int32)
    for ba in range(batchSize):
        for (x_pos, y_pos, state) in alignments[ba]:
            if state < 3:
                obs_Alignments[ba, x_pos, y_pos, state] = 1
    return obs_Alignments


# update the Lambda if necessary
# lambda use subgradient descent as:
#     lambda^{n+1} = lambda^n - rho * (z - y)
def update_lambda(Lambda, alignment_y, alignment_z, Rho=0.01):
    assert len(alignment_y) == len(alignment_z), "batchSize should equal"
    batchSize = len(alignment_y)

    for ba in range(batchSize):
        # update alignment_z
        alignLenZ = alignment_z[ba].size(0)
        for i in range(alignLenZ):
            x_pos, y_pos, state = alignment_z[ba][i]
            if state < 3:
                Lambda[ba, x_pos, y_pos, state] -= Rho

        # update alignment_y
        alignLenY = alignment_y[ba].size(0)
        for j in range(alignLenY):
            x_pos, y_pos, state = alignment_y[ba][j]
            if state < 3:
                Lambda[ba, x_pos, y_pos, state] += Rho

    return Lambda


# for a search space in x_pos,
# return the search space larger or equal / smaller than y_pos
# for a sorted indices, use bisect for O(log n) time complexity
def ModifySpace(space, x_pos, xLen, larger):
    x_pos = float(x_pos)
    if larger:
        space_filter = torch.ge(space, min(xLen, x_pos+6)).long()
        search_space = space_filter * space
    else:
        space_filter = torch.lt(space, max(0, x_pos-6)).long()
        search_space = space_filter * space
    return search_space, space_filter.float()
