import torch
import torch.nn as nn
import numpy as np
import sharedmem
from torch.autograd import Function
from . import interface as cpp
from .Utils import DetectMultiHIS
from .Utils import Alignment_Singleton_Score, getStateNum, getIdenticals, \
    getAlignmentStart, getAlignmentEnd


class CRFLoss(nn.Module):
    def __init__(self, numStates, device):
        super(CRFLoss, self).__init__()
        self.numStates = numStates
        self.device = device
        self.P = nn.Parameter(torch.zeros(size=(5, 5), device=device))
        self.trans = initialize_transitions(numStates, device=device)

    def reset_parameters(self):
        self.trans = initialize_transitions(self.numStates, device=self.device)

    def forward(self, observations, alignments, maskX, maskY,
                pool_forward, pool_backward, pool_score, auto_grad=True):
        batchSize, xLen, yLen, _ = observations.size()
        observations = observations.to(self.device)
        alignments = alignments.to(self.device)
        mask = torch.LongTensor([[maskX[ba], maskY[ba]] for
                                 ba in range(batchSize)])
        if auto_grad:
            return self._forward(observations, alignments, mask)
        else:
            return CRFF.apply(observations, self.trans, self.P,
                              alignments, mask,
                              pool_forward, pool_backward, pool_score)

    def _forward(self, observations, alignments, mask):
        # caculate the negative log likelihood
        # negatice log likelihood : - log(p(y|X)) is the loss function
        # - log (p(y|X)) = partition - aliScore for corresponding alignments
        # partition = Zf + Zb / 2
        transitions = self.trans + self.P
        obs = observations.detach().numpy()
        tra = transitions.detach().numpy()
        mas = np.array(mask).astype(np.int)

        Zf, alpha = cpp.crf_batchforward(obs, tra, mas)
        Zb, beta = cpp.crf_batchbackward(obs, tra, mas)

        partition = (Zf + Zb) / 2
        partition = torch.from_numpy(partition)
        aliScore = BatchScoring(observations, alignments, transitions)
        neg_log_likelihood = partition - aliScore

        return neg_log_likelihood


# calculate the log marginal probability of a vertex (ba, i, j, s) in the
# alignment path use alpha[ba, i, j, s] + beta[ba, i, j, s] - \
#        observations[ba, i, j, s] to caculate the marginal matrix
# observations shall be subtracted since it is included in both alpha and beta
# to get the true marginal probability, we shall substract partition
def marginal(vertex, observations, alpha, beta, partition=0):
    return alpha[tuple(vertex)] + beta[tuple(vertex)] - \
            observations[tuple(vertex)] - partition


def marginalMatrix(observations, alpha, beta, partition, mask):
    batchSize, xLen, yLen, numStates = observations.size()
    partition = partition.unsqueeze(-1).unsqueeze_(-1).unsqueeze_(-1).expand(
            batchSize, xLen, yLen, numStates)
    marginal = alpha + beta - observations - partition
    for ba in range(batchSize):
        marginal[ba, mask[ba][0]:, mask[ba][1]:, :].fill_(float('-inf'))
    marginal = torch.exp(marginal)

    return marginal


def CalcMarginal(observations, transitions):
    batchSize, xLen, yLen, _ = observations.size()
    mask = torch.LongTensor([[xLen, yLen]] * batchSize)
    mask = mask.numpy()

    partition1, alpha = _crf_BatchForward(observations, transitions, mask)
    partition2, beta = _crf_BatchBackward(observations, transitions, mask)

    partition = (partition1 + partition2) / 2
    Marginal_score = marginalMatrix(observations, alpha, beta,
                                    partition, mask)
    return Marginal_score


# calculate the \frac{\partial{logP(y|x)}}{\partial{u_{y,y'}}}
# sigma f_k(y_{i-1}, y_{i}, x)p(y_{i-1}, y_{i}|x;)
# return the I alignment matrix to caculate the gradient of observations
# the shape of alignments is (batchSize, alignLen, 3)
# we caculate the new shape (batchSize, xLen, yLen, 3),
# if it is exist the pair, the point will be 1, otherwise it will be 0
def obsAlignment(alignments, observations):
    batchSize, xLen, yLen, numStates = observations.size()
    prefix = torch.zeros((batchSize, alignments.size(1), 1),
                         dtype=torch.long, device=alignments.device)
    for i in range(1, batchSize):
        prefix[i].fill_(i)
    new_alignments = torch.cat([prefix, alignments.long()],
                               dim=-1).long().view(-1, 4)
    obs_Alignments = torch.zeros(observations.size(), dtype=torch.float,
                                 device=observations.device)
    # alignment start on 1
    for item in new_alignments:
        ba, x_pos, y_pos, state = item
        if state < 3 and (x_pos != 0 and y_pos != 0):
            # match, insertX or insertY
            obs_Alignments[ba, x_pos-1, y_pos-1, state] += 1
    return obs_Alignments


# here, alignments is a tensor with shape(BatchSize, AlignLen, 3)
# and the alignments starts 1, rather than 0
# for tail and head gap, the alignment maybe (x,y,3) or (x,y,4)
# for padding position, the alignment maybe (0, 0, 5)
def BatchScoring(obs, alignments, transitions):
    # check the valid of the data
    observations = torch.from_numpy(obs)
    batchSize, xLen, yLen, numStates = observations.size()
    # assert transitions.size() == (numStates+2, numStates+2)
    new_observations = torch.zeros([batchSize, xLen+1, yLen+1, numStates+3])
    new_observations[:, 1:, 1:, :3] = observations

    # add the batchNum in alignments
    alignLength = alignments.size(1)
    alignments = alignments.to(observations.device)
    prefix = torch.zeros((batchSize, alignLength, 1), dtype=torch.long,
                         device=observations.device)
    for i in range(1, batchSize):
        prefix[i].fill_(i)
    new_alignments = torch.cat([prefix, alignments], dim=-1)
    # new_alignments has shape (batchSize, alignLength, 4)

    obsScore = new_observations[
            tuple(new_alignments.permute(2, 0, 1).view(
                4, batchSize*alignLength))].view(batchSize, alignLength)
    new_transition = torch.zeros(batchSize, numStates+3, numStates+3)
    new_transition[:, :numStates+2, :numStates+2] = transitions

    transScore = torch.zeros(batchSize, alignLength-1)
    for ba in range(batchSize):
        prevState = alignments[ba, :-1, -1]
        currState = alignments[ba, 1:, -1]
        transScore[ba] = new_transition[ba, prevState, currState]

    # scores has shape (batchSize)
    scores = torch.sum(obsScore, dim=-1) + torch.sum(transScore, dim=-1)

    return scores


def _crf_BatchForward(obs, trans, mask):
    tra = trans.detach().numpy()
    mas = np.array(mask).astype(np.int)
    Zf, alpha = cpp.crf_batchforward(obs, tra, mas)
    Zf = torch.from_numpy(Zf).float()
    alpha = torch.from_numpy(alpha).float()

    return Zf, alpha


def _crf_BatchBackward(obs, trans, mask):
    tra = trans.detach().numpy()
    mas = np.array(mask).astype(np.int)
    Zb, beta = cpp.crf_batchbackward(obs, tra, mas)
    Zb = torch.from_numpy(Zb).float()
    beta = torch.from_numpy(beta).float()

    return Zb, beta


# add head and tail gap to an alignment. alignment is a local aligment for
# two sequences with length xLen and yLen
# alignment may not contain head and tail gap information. ExpandAlignment
# will add head and tail gap information to form a full alignment
def ExpandAlignment(alignment_, xLen, yLen):
    alignment = torch.tensor(alignment_, dtype=torch.int32)
    if alignment.size()[0] == 0:
        # when alignment is empty
        fullAlignment = [(i, -1, 3) for i in range(1, xLen+1)] +\
                [(-1, j, 3) for j in range(1, yLen+1)]
        return torch.tensor(fullAlignment)

    # when alignment is not empty
    hGapPositions = alignment[0]
    tGapPositions = alignment[-1]
    fullAlignment = [(i, -1, 3) for i in range(0, hGapPositions[0])] + \
                    [(-1, j, 3) for j in range(0, hGapPositions[1])]
    fullAlignment.extend(alignment.tolist())
    fullAlignment.extend([(i, -1, 4) for i in
                         range(tGapPositions[0] + 1, xLen)])
    fullAlignment.extend([(-1, j, 4) for j in
                         range(tGapPositions[1] + 1, yLen)])

    return torch.LongTensor(fullAlignment)


class CRFF(Function):
    @staticmethod
    def forward(self, observations, trans, P, alignments, mask,
                pool_forward, pool_backward, pool_score):
        # caculate the partition function to caculate the neg log likelihood
        # caculate the log pos for the alignment by BatchScoring
        # neg_log_likelihood = partition - aliScore
        transitions = trans + P
        batchSize = observations.size(0)
        transitions = transitions.expand(batchSize, 5, 5)
        obs = np.array(observations.detach())
        sharedobs = sharedmem.empty(obs.size)
        sharedobs = obs

        forward_async = pool_forward.apply_async(
                _crf_BatchForward,
                (sharedobs, transitions.detach(), mask))
        backward_async = pool_backward.apply_async(
                _crf_BatchBackward,
                (sharedobs, transitions.detach(), mask))
        score_async = pool_score.apply_async(
                BatchScoring,
                (sharedobs, alignments, transitions.detach()))

        # we caculate the alpha and beta parallel
        Zf, alpha = forward_async.get()
        Zb, beta = backward_async.get()
        aliScore = score_async.get()
        # print("Zf - Zb:", Zf - Zb)
        partition = (Zf + Zb) / 2

        self.save_for_backward(observations, transitions, alignments,
                               partition, alpha, beta, mask)
        neg_log_likelihood = partition - aliScore

        marginal = marginalMatrix(observations, alpha, beta, partition, mask)
        marginalScore = marginalloss(marginal, alignments)

        return neg_log_likelihood, marginalScore

    @staticmethod
    def backward(self, crfLoss, marginalLoss):
        # caculate the gradient for the loss Function to updata the Parameter
        # we use neg log likelihood
        # so the gradient is marginalMatrix - obsAlignment
        observations, transitions, alignments, partition, alpha, beta, mask \
                = self.saved_tensors
        marginalObs = marginalMatrix(
                observations.detach(), alpha, beta, partition, mask)
        realObs = obsAlignment(alignments, observations.detach())

        obsgrad = marginalObs - realObs

        return obsgrad, None, None, None, None, None, None, None


# get the MaxAcc score, to evaluate the model
def marginalloss(marginal, alignments):
    batchSize, alignLen, _ = alignments.size()
    marginalScore = torch.zeros((batchSize,), device=marginal.device)

    for ba in range(batchSize):
        realLen = 0
        for i in range(alignLen):
            x_pos, y_pos, state = alignments[ba][i]
            # only consider match
            if state == 0:
                realLen += 1
                marginalScore[ba] += marginal[ba, x_pos-1, y_pos-1, state]
        if realLen == 0:
            marginalScore[ba] = 0
        else:
            marginalScore[ba] = marginalScore[ba] / realLen

    return marginalScore


# use MaxAcc as objective function, and use viterbi to find a alignment
def BatchViterbi4alignment(potential, tpl, tgt):
    penalty = torch.Tensor([[0, 0, 0, float('-inf')],
                            [0, 0, 0, float('-inf')],
                            [0, float('-inf'), 0, float('-inf')],
                            [0, float('-inf'), float('-inf'), 0]])
    return viterbi(potential, penalty, tpl, tgt, MODIFY=False)


def viterbi(observation, transition, tpl, tgt, MODIFY=False):
    batchSize, xLen, yLen, numStates = observation.size()
    obs = observation.detach().numpy()
    tra = transition.detach().numpy()
    masX = np.array([xLen]*batchSize).astype(np.int)
    maxScores, argmaxPos, tracebacks = cpp.viterbi(obs, tra, masX)
    maxScores = torch.from_numpy(maxScores)
    argmaxPos = torch.from_numpy(argmaxPos)
    tracebacks = torch.from_numpy(tracebacks)
    alignments = []
    for score_max, pos_max, traceback in zip(maxScores, argmaxPos, tracebacks):
        if score_max <= 0.0:
            alignments.append(ExpandAlignment([], xLen, yLen))
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

        alignment = ExpandAlignment(pos_list, xLen, yLen)
        if MODIFY:
            alignment = ModifyAlignment(alignment, observation, tpl, tgt)
        alignments.append(alignment)

    return maxScores, alignments


def realalignLen(alignments):
    real = 0
    for (x_pos, y_pos, state) in alignments:
        if state == 0:
            real += 1
    return real


# deal with missing for observation
def ModifyObs(tpl, observations):
    batchSize, xLen, yLen, _ = observations.size()
    isMissing = torch.einsum('a,b->ab',
                             torch.Tensor(tpl['missing']).bool(),
                             torch.ones(yLen).bool())
    obs_filter = isMissing
    obs_filter = obs_filter.unsqueeze(0).unsqueeze(-1).expand(
        batchSize, xLen, yLen, 3)
    observations = torch.where(obs_filter, observations - 5, observations)
    return observations


# deal with bad and missing
def ModifyAlignment(alignment, observations, tpl, tgt):
    AlignLen = alignment.size(0)
    TrueAlign = []
    for i in range(AlignLen):
        badcheck = 0
        x_pos, y_pos, state = alignment[i]
        if state == 0:
            if tpl['sequence'][x_pos] == "X" or tgt['sequence'][y_pos] == "X" \
                    or (tpl['missing'][x_pos] == 1 and
                        tpl['sequence'][x_pos] != tgt['sequence'][y_pos]):
                badcheck = 1
            if DetectMultiHIS(tpl['sequence'])[x_pos] == 1 or \
                    DetectMultiHIS(tgt['sequence'])[y_pos] == 1:
                badcheck = 1
            # if observations[0, x_pos, y_pos, 0] < 0:
            #     badcheck = 1
            if badcheck == 1:
                if x_pos != 0:
                    TrueAlign.append([x_pos-1, y_pos, 1])
                    TrueAlign.append([x_pos, y_pos, 2])
                else:
                    TrueAlign.append([0, -1, 3])
                    TrueAlign.append([-1, 0, 3])
            else:
                TrueAlign.append([x_pos, y_pos, state])
        else:
            TrueAlign.append([x_pos, y_pos, state])

    if realalignLen(torch.LongTensor(TrueAlign)) == 0:
        return alignment

    # fix the head
    for i in range(len(TrueAlign)):
        if TrueAlign[i][2] != 0:
            if TrueAlign[i][2] != 3:
                x_pos, y_pos, state = TrueAlign[i]
                TrueAlign[i] = [x_pos, -1, 3] if state == 1 else [-1, y_pos, 3]
        else:
            break

    # fix the tail
    for i in range(len(TrueAlign)-1, -1, -1):
        if TrueAlign[i][2] != 0:
            if TrueAlign[i][2] != 4:
                x_pos, y_pos, state = TrueAlign[i]
                TrueAlign[i] = [x_pos, -1, 4] if state == 1 else [-1, y_pos, 4]
        else:
            break

    return torch.LongTensor(TrueAlign)


# for local alignment, head and tail gap penalty is set to 0
# the transition is important, it will determine the performance of threading.
def initialize_transitions(numStates=3, device=None):
    assert (numStates == 3), \
            "currently only 3 alignment states (match, insert in X and \
            insert in Y) plus head and tail gaps are supported"
    trans = torch.zeros((numStates+2, numStates+2),
                        dtype=torch.float, device=device)

    #                            match insertX insertY headGap tailGap
    trans[0] = torch.FloatTensor([0.5, -5, -5, float('-inf'), 0.])
    trans[1] = torch.FloatTensor([0, -1, -2, float('-inf'), float('-inf')])
    trans[2] = torch.FloatTensor([0, float('-inf'), -1, float('-inf'),
                                  float('-inf')])
    trans[3] = torch.FloatTensor([0.0, float('-inf'), float('-inf'), 0.,
                                  float('-inf')])
    trans[4] = torch.FloatTensor([float('-inf'), float('-inf'), float('-inf'),
                                 float('-inf'), 0.])

    return trans


def initialize_penalty(numStates=3, device=None):
    assert (numStates == 3), \
            "currently only 3 alignment states (match, insert in X and \
            insert in Y) plus head and tail gaps are supported"
    penalty = torch.Tensor([[0, 0, 0, float('-inf'), 0],
                            [0, 0, 0, float('-inf'), float('-inf')],
                            [0, float('-inf'), 0, float('-inf'),
                             float('-inf')],
                            [0, float('-inf'), float('-inf'), 0,
                             float('-inf')],
                            [float('-inf'), float('-inf'), float('-inf'),
                             float('-inf'), 0]])
    return penalty


def MaxAcc_init(obs, trans, maskX, maskY, tpldata, tgtseq):
    observations = torch.from_numpy(obs)
    batchSize, xLen, yLen, _ = observations.size()
    mask = torch.LongTensor([[maskX[ba], maskY[ba]] for
                            ba in range(batchSize)])

    partition1, alpha = _crf_BatchForward(obs, trans, mask)
    partition2, beta = _crf_BatchBackward(obs, trans, mask)

    partition = (partition1 + partition2) / 2
    singleton_score = marginalMatrix(observations, alpha, beta,
                                     partition, mask)
    sing = singleton_score.detach().numpy().astype(np.float32)

    penalty = torch.Tensor([[0, 0, 0, float('-inf'), 0],
                            [0, 0, 0, float('-inf'), float('-inf')],
                            [0, float('-inf'), 0, float('-inf'),
                             float('-inf')],
                            [0, float('-inf'), float('-inf'), 0,
                             float('-inf')],
                            [float('-inf'), float('-inf'), float('-inf'),
                             float('-inf'), 0]])
    pen = penalty.expand(batchSize, 5, 5).detach().numpy().astype(np.float32)

    maxScores, argmaxPos, tracebacks = cpp.viterbi(
            sing, pen, np.array(maskX).astype(np.int))

    return maxScores, argmaxPos, tracebacks


def get_CNF_output(tpl, tgt, alignments, observations, transitions,
                   method="Viterbi"):
    if method == "Viterbi":
        node_score = Alignment_Singleton_Score(
                [alignments], observations, transitions)[0]
    elif method == "MaxAcc":
        batchSize = observations.size(0)
        penalty = initialize_penalty(3)
        penalty = penalty.expand(batchSize, 5, 5)
        node_score = Alignment_Singleton_Score(
                [alignments], observations, penalty)[0]
    queryname = tgt['name']
    subjectname = tpl['name']
    query_length = tgt['length']
    subject_length = tpl['length']
    align_col, querygap, subjectgap = getStateNum(alignments)
    query_start, subject_start = getAlignmentStart(alignments)
    query_end, subject_end = getAlignmentEnd(alignments)
    alignment_score = node_score
    identicals = getIdenticals(alignments, tpl['sequence'], tgt['sequence'])
    SeqId = identicals / min(query_length, subject_length) * 100
    output = '{:<14s}{:<12s}{:<5d}{:<5d}'\
             '{:<5d}{:<5d}{:<5d}{:<5d}'\
             '{:<5d}{:<5d}{:<5d}{:<10.2f}'\
             '{:<5d}{:<10.2f}\n'.format(
              queryname, subjectname, query_length,
              subject_length, align_col, querygap, subjectgap,
              query_start, query_end, subject_start,
              subject_end, alignment_score, identicals, SeqId)
    return output
