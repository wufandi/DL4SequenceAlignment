import os
import sys
import torch
import pickle
import numpy as np

MATCH_BOUND_UPPER = 15
MATCH_BOUND_LOWER = -5

# conditional distribution p(d_{ik}^T|d_{jl}^S) for 12 bins,
# we can get joint distribution by multiplying p_ref(d_{jl}^S)
# background distribution p_ref(d_{ik}^T) for 12 bins:
# <5, 5~6, ..., 14~15, >=15
edge_potent_d_ref = torch.Tensor(
        [0.001401900, 0.003644220, 0.004355920, 0.004943320,
         0.008613160, 0.015290900, 0.021347600, 0.019730100,
         0.022954800, 0.024599900, 0.028235800, 0.844883000])


# return a tensor with Len * Len * 12 shape
# represent the probility of x_pos and x_pos distance in 12 different intervals
def Load_EPAD(tpl_epad_source, length):
    # check file exist
    if os.path.exists(tpl_epad_source):
        pair_dis = np.zeros((length, length, 12))
        distance_file = open(tpl_epad_source, 'r')
        for line in distance_file:
            data = line.strip('\n').split()
            index1 = int(data[0])
            index2 = int(data[1])
            dis_prob = np.array(list(map(float, data[2:])))
            np.putmask(dis_prob, dis_prob == 0, 0.01)
            pair_dis[index1, index2, :12] = 0.01 * dis_prob[:12]
            pair_dis[index2, index1, :12] = 0.01 * dis_prob[:12]
        distance_file.close()
        disc_method = [0.0] + np.linspace(5.0, 15.0, num=11).tolist()
        edge_type = "epad"
        return pair_dis, disc_method, edge_type
    else:
        print("epad_file %s is not found !" % (tpl_epad_source))
        sys.exit(-1)


# load the distance potential file,
# return the sequence name, sequence, disc_potential and disc_method
def Load_DisPotential(path, tgt):
    if not os.path.exists(path):
        print("distance potential %s is not found" % path)
        sys.exit(-1)
    with open(os.path.join(path), 'rb') as W:
        name, seq, dis_pot, disc_method = pickle.load(
                W, encoding='latin1')[:4]
        score_type = list(disc_method)[0]
        assert seq == tgt['sequence'], \
            "the seq of dist_pot:\n%s should equal to query seq \n%s" % \
            (seq, tgt['sequence'])
        assert name == tgt['name'], \
            "the name of dist_pot:\n%s should equal to query seq \n%s" % \
            (name, tgt['name'])
        edge_type = "dist"

    return dis_pot[score_type], disc_method[score_type], edge_type


# load the edge score file,
# only support epad and distance potential
def Load_EdgeScore(path, tgt):
    if path.endswith(".epad_prob"):
        return Load_EPAD(path, tgt['length'])
    else:
        return Load_DisPotential(path, tgt)


# map the distance to index
def map_distances2index(dis, disc_method):
    dis = dis.numpy()
    index = np.searchsorted(disc_method, dis)
    index = torch.from_numpy(index)
    return index


# calculate edge potential for a pair of aligned position
# ti, tj: template i+1/j+1 residue, qi, qj query i+1/j+1 residue i,j=0,1,2...
# dis_matrix has shape (Len, Len)
def calculate_edge_potential(ti, tj, qi, qj, pair_dis, dis_matrix,
                             disc_method, edge_type="dist", Edge_Norm=1):
    pot = 0
    dis = dis_matrix[ti, tj].numpy()
    t_index = np.searchsorted(disc_method, dis)
    if edge_type == "dist":
        pot = -pair_dis[qi, qj, t_index]
    elif edge_type == "epad":
        pot = torch.log(pair_dis[qi, qj, t_index] / edge_potent_d_ref[t_index])
        pot = pot.item()
        if pot > MATCH_BOUND_UPPER:
            pot = MATCH_BOUND_UPPER
        elif pot < MATCH_BOUND_LOWER:
            pot = MATCH_BOUND_LOWER
    else:
        print("we only support dist and epad type for edge score")
        sys.exit(-1)
    pot = pot / Edge_Norm
    return pot


# calculate the edge potential matrixlizely
# input:
#  tensor4tpl, tpl_pos, tensor4tgt, tgt_pos,
#  pair_dis, dis_matrix, disc_method
def calculate_edge_potential_matrix_X3(
        Ti, tj, Qi, qj, pair_dis, dis_matrix, disc_method,
        edge_type="dist", Edge_Norm=1):
    Ti = Ti.view(-1)
    t_index = map_distances2index(dis_matrix[Ti, tj], disc_method)
    if edge_type == "dist":
        pot = -pair_dis[Qi, qj, t_index]
    elif edge_type == "epad":
        pot = torch.log(pair_dis[Qi, qj, t_index] / edge_potent_d_ref[t_index])
        pot = torch.clamp(pot, min=MATCH_BOUND_LOWER, max=MATCH_BOUND_UPPER)
    else:
        print("we only support dist and epad type for edge score")
        sys.exit(-1)
    pot = pot / Edge_Norm
    return pot


# calculate the edge potential matrixlizely
# input:
#   tgt_pos, tensor4tgt, tpl_pos, tensor4tpl,
#   pair_dis, dis_matrix, disc_method
def calculate_edge_potential_matrix_X4(
        ti, Tj, qi, Qj, pair_dis, dis_matrix, disc_method,
        edge_type="dist", Edge_Norm=1):
    Tj = Tj.view(-1)
    t_index = map_distances2index(dis_matrix[ti, Tj], disc_method)
    if edge_type == "dist":
        pot = -pair_dis[qi, Qj, t_index]
    elif edge_type == "epad":
        pot = torch.log(pair_dis[qi, Qj, t_index] / edge_potent_d_ref[t_index])
        pot = torch.clamp(pot, min=MATCH_BOUND_LOWER, max=MATCH_BOUND_UPPER)
    else:
        print("we only support dist and epad type for edge score")
        sys.exit(-1)
    pot = pot / Edge_Norm
    return pot
