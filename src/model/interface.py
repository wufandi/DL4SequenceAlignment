import numpy as np
import os
import ctypes
import sys

_path = os.path.dirname('__file__')
if ".." in sys.path:
    _path = ".."
lib_crf = np.ctypeslib.load_library('src/model/crf', _path)
lib_dist = np.ctypeslib.load_library('src/model/distance', _path)

_Crftypedict = {'viterbi_all': float,
                'crf_forward': float,
                'crf_backward': float}
_Distypedict = {'compute_distance_score': float,
                'compute_distance': float}


for name in _Crftypedict.keys():
    val = getattr(lib_crf, name)
    if name == 'viterbi_all':
        val.argtypes = [np.ctypeslib.ndpointer(float,
                                               flags='aligned, contiguous'),
                        np.ctypeslib.ndpointer(float,
                                               flags='aligned, contiguous'),
                        np.ctypeslib.c_intp,
                        np.ctypeslib.c_intp,
                        np.ctypeslib.c_intp,
                        np.ctypeslib.ndpointer(int,
                                               flags='aligned, contiguous,'
                                               'writeable'),
                        np.ctypeslib.ndpointer(int,
                                               flags='aligned, contiguous,'
                                               'writeable')]
        val.restype = ctypes.c_double

    elif name == 'crf_forward' or name == 'crf_backward':
        val.argtypes = [np.ctypeslib.ndpointer(float,
                                               flags='aligned, contiguous'),
                        np.ctypeslib.ndpointer(float,
                                               flags='aligned, contiguous'),
                        np.ctypeslib.c_intp,
                        np.ctypeslib.c_intp,
                        np.ctypeslib.c_intp,
                        np.ctypeslib.c_intp,
                        np.ctypeslib.ndpointer(float,
                                               flags='aligned, contiguous,'
                                               'writeable')]
        val.restype = ctypes.c_double

for name in _Distypedict.keys():
    val = getattr(lib_dist, name)
    if name == 'compute_distance_score':
        val.argtypes = [np.ctypeslib.ndpointer(int,
                                               flags='aligned, contiguous'),
                        np.ctypeslib.ndpointer(float,
                                               flags='aligned, contiguous'),
                        np.ctypeslib.ndpointer(float,
                                               flags='aligned, contiguous'),
                        np.ctypeslib.ndpointer(float,
                                               flags='aligned, contiguous'),
                        ctypes.c_double,
                        np.ctypeslib.c_intp,
                        np.ctypeslib.c_intp,
                        np.ctypeslib.c_intp,
                        np.ctypeslib.c_intp,
                        np.ctypeslib.c_intp,
                        np.ctypeslib.ndpointer(float,
                                               flags='aligned, contiguous,'
                                               'writeable')]
        val.restype = None

    elif name == 'compute_distance':
        val.argtypes = [ctypes.c_char_p,
                        np.ctypeslib.c_intp,
                        np.ctypeslib.ndpointer(int,
                                               flags='aligned, contiguous'),
                        np.ctypeslib.ndpointer(float,
                                               flags='aligned, contiguous'),
                        np.ctypeslib.ndpointer(float,
                                               flags='aligned, contiguous'),
                        np.ctypeslib.ndpointer(float,
                                               flags='aligned, contiguous,'
                                               'writeable')]
        val.restype = None


def viterbi(observation, transition, maskX):
    requires = ['CONTIGUOUS', 'ALIGNED', 'C_CONTIGUOUS']
    batchSize, xLen, yLen, _ = observation.shape
    maxScore = np.zeros((batchSize))
    argmaxPos = np.zeros((batchSize, 2), dtype=int)
    traceback = np.zeros((batchSize, xLen, yLen, 3), dtype=int)
    trans = np.asanyarray(transition, dtype=float)
    trans = np.require(trans, float, requires)

    for ba in range(batchSize):
        tpl_length = maskX[ba]
        obs = np.asanyarray(observation[ba], dtype=float)
        obs = np.require(obs, float, requires)
        maxScore[ba] = lib_crf.viterbi_all(
            obs, trans[ba], xLen, yLen, tpl_length,
            argmaxPos[ba], traceback[ba])
    return maxScore, argmaxPos, traceback


def crf_batchforward(observation, transition, mask):
    requires = ['CONTIGUOUS', 'ALIGNED', 'C_CONTIGUOUS']
    batchSize, xLen, yLen, _ = observation.shape
    partition = np.zeros((batchSize))
    alpha = np.zeros((batchSize, xLen, yLen, 3), dtype=float)
    trans = np.asanyarray(transition, dtype=float)
    trans = np.require(trans, float, requires)

    for ba in range(batchSize):
        tpl_length, tgt_length = mask[ba]
        obs = np.asanyarray(observation[ba], dtype=float)
        obs = np.require(obs, float, requires)
        partition[ba] = lib_crf.crf_forward(
            obs, trans[ba], xLen, yLen, tpl_length, tgt_length, alpha[ba])
    return partition, alpha


def crf_batchbackward(observation, transition, mask):
    requires = ['CONTIGUOUS', 'ALIGNED', 'C_CONTIGUOUS']
    batchSize, xLen, yLen, _ = observation.shape
    partition = np.zeros((batchSize))
    beta = np.zeros((batchSize, xLen, yLen, 3), dtype=float)
    trans = np.asanyarray(transition, dtype=float)
    trans = np.require(trans, float, requires)

    for ba in range(batchSize):
        tpl_length, tgt_length = mask[ba]
        obs = np.asanyarray(observation[ba], dtype=float)
        obs = np.require(obs, float, requires)
        partition[ba] = lib_crf.crf_backward(
            obs, trans[ba], xLen, yLen, tpl_length, tgt_length, beta[ba])
    return partition, beta


def compute_pairwise_score(alignment, pair_dis, dis_matrix, disc_method,
                           Norm_Weight, use_dist_pot=True):
    requires = ['CONTIGUOUS', 'ALIGNED', 'C_CONTIGUOUS']
    alignLen = alignment.shape[0]
    xLen = dis_matrix.shape[0]
    yLen = pair_dis.shape[0]
    method_length = disc_method.shape[0]

    alignment = np.asanyarray(alignment, dtype=int)
    alignment = np.require(alignment, int, requires)

    pair_dis = np.asanyarray(pair_dis, dtype=float)
    pair_dis = np.require(pair_dis, float, requires)

    dis_matrix = np.asanyarray(dis_matrix, dtype=float)
    dis_matrix = np.require(dis_matrix, float, requires)

    disc_method = np.asanyarray(disc_method, dtype=float)
    disc_method = np.require(disc_method, float, requires)

    if use_dist_pot:
        dist_option = 1
    else:
        dist_option = 0

    score = np.zeros((2), dtype=float)

    lib_dist.compute_distance_score(
        alignment, pair_dis, dis_matrix, disc_method, Norm_Weight,
        dist_option, xLen, yLen, alignLen, method_length, score)
    distance_score, norm_score = score
    return distance_score, norm_score


def compute_distance(sequence, tpl_length, missing, CA, CB):
    requires = ['CONTIGUOUS', 'ALIGNED', 'C_CONTIGUOUS']
    b_string = sequence.encode('utf-8')

    missing = np.asanyarray(missing, dtype=int)
    missing = np.require(missing, int, requires)

    CA = np.asanyarray(CA, dtype=float)
    CA = np.require(CA, float, requires)

    CB = np.asanyarray(CB, dtype=float)
    CB = np.require(CB, float, requires)

    dis_matrix = np.zeros((tpl_length, tpl_length), dtype=float)
    lib_dist.compute_distance(
        b_string, tpl_length, missing, CA, CB, dis_matrix)
    return dis_matrix
