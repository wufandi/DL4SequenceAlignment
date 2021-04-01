import os
import sys
import torch
import math
from multiprocessing import Pool
from . import FeatureCreate
from .LoadTPLTGT import load_tpl
from .SequenceUtils import AALetter2OrderOf1LetterCode, \
    AALetter2OrderOf3LetterCode


# read the data from list file, store them in one list
def readDataList(train_list, number=0):
    dataset = open(train_list, 'r')
    seqdata = []
    line_nu = 0
    for line in dataset.readlines():
        if number != 0:
            # for test now
            if line_nu < number:
                data = line.strip('\n')
                seqdata.append(data)
                line_nu += 1
            else:
                break
        else:
            # all dataset
            data = line.strip('\n')
            seqdata.append(data)
            line_nu += 1
    dataset.close()
    return seqdata


# used to sepreate the template data
# the output must be a list of template_list,
# templat_list is a list of template_names
# the template_length is estimated based on the file size
# maxx is the template length limit, default is 1000
# stride is the classification of different size,
# cpu is the number of cpu we use to speed up to estimate template length,
# default is 20
# gpu is the number of gpu we use, it will affect the size of mini-batch
def TemplateSepreate(template_list, tpl_root, tpllimit=1000,
                     batchlimit=8, stride=1, cpu=20, gpu=1):
    tpl_length_list = generate_tplLengthfromList(template_list, tpl_root, cpu)

    lengClass = [[] for x in range(math.ceil(tpllimit/stride))]
    batchData = []

    for tpl in tpl_length_list:
        tplName, tplLength = tpl
        if tplLength >= tpllimit:
            batchData.append([tplName])
        else:
            lengClass[tplLength//stride].append(tplName)

    for x_index in range(math.ceil(tpllimit/stride)):
        if lengClass[x_index] == [[]]:
            continue
        PERBATCH = min(batchlimit,
                       max(gpu * tpllimit // ((x_index+1)*stride), gpu))

        for i in range(0, len(lengClass[x_index]), PERBATCH):
            if lengClass[x_index][i:i+PERBATCH] != []:
                batchData.append(lengClass[x_index][i:i+PERBATCH])
    return batchData


# used to create the feature for a sequence ,
# we generate the following 10 initial similarity features
# without structure information
# seq_id, blosum80, blosum62, blosum45, spScore,
# spScore_ST, pmScore, pmScore_ST, cc, hdsm
# then, we generate ss3, ss8, acc by different FeatureMode
def feature4sequence(tpl, tgt, SS3FeatureMode, SS8FeatureMode, ACCFeatureMode):
    xLen = tpl['length']
    yLen = tgt['length']
    featSize = 10 + SS3FeatureMode + SS8FeatureMode + ACCFeatureMode
    feature = torch.zeros((1, xLen, yLen, featSize))
    tplseq = tpl['sequence']
    tgtseq = tgt['sequence']
    tpl1LetterCode = list(map(AALetter2OrderOf1LetterCode.get,
                          list(tplseq)))
    tpl3LetterCode = list(map(AALetter2OrderOf3LetterCode.get,
                          list(tplseq)))
    tgt1LetterCode = list(map(AALetter2OrderOf1LetterCode.get,
                          list(tgtseq)))
    tgt3LetterCode = list(map(AALetter2OrderOf3LetterCode.get,
                          list(tgtseq)))

    feature[0, :, :, 0] = torch.as_tensor(
            FeatureCreate.seq_id(tpl1LetterCode, tgt1LetterCode))
    feature[0, :, :, 1] = torch.as_tensor(
            FeatureCreate.blosum80(tpl3LetterCode, tgt3LetterCode))
    feature[0, :, :, 2] = torch.as_tensor(
            FeatureCreate.blosum62(tpl3LetterCode, tgt3LetterCode))
    feature[0, :, :, 3] = torch.as_tensor(
            FeatureCreate.blosum45(tpl3LetterCode, tgt3LetterCode))
    feature[0, :, :, 4] = torch.as_tensor(
            FeatureCreate.mutationof2pos6(
                tpl1LetterCode, tgt1LetterCode,
                tpl['PSSM'], tgt['PSSM']))
    feature[0, :, :, 5] = torch.as_tensor(
            FeatureCreate.mutationof2pos6_st(
                tpl1LetterCode, tgt1LetterCode, tpl['PSSM']))
    feature[0, :, :, 6] = torch.as_tensor(
            FeatureCreate.mutationof2pos5(
                tpl['PSSM'], tpl['PSFM'],
                tgt['PSSM'], tgt['PSFM']))
    feature[0, :, :, 7] = torch.as_tensor(
            FeatureCreate.mutationof2pos5_st(tpl['PSSM'], tgt['PSFM']))
    feature[0, :, :, 8] = torch.as_tensor(
            FeatureCreate.cc50(tpl3LetterCode, tgt3LetterCode))
    feature[0, :, :, 9] = torch.as_tensor(
            FeatureCreate.hdsm(tpl3LetterCode, tgt3LetterCode))
    # index start from 10
    index = 10
    if SS3FeatureMode != 0:
        feature[0, :, :, index:index+SS3FeatureMode] = torch.as_tensor(
                FeatureCreate.ss3(tpl['SS3Coding'], tgt['SS3'],
                                  SS3FeatureMode))
    index += SS3FeatureMode
    if SS8FeatureMode != 0:
        feature[0, :, :, index:index+SS8FeatureMode] = torch.as_tensor(
                FeatureCreate.ss8(tpl['SS8Coding'], tgt['SS8'],
                                  SS8FeatureMode))
    index += SS8FeatureMode
    if ACCFeatureMode != 0:
        feature[0, :, :, index:index+ACCFeatureMode] = torch.as_tensor(
                FeatureCreate.acc(tpl['ACC'], tgt['ACC_prob'],
                                  ACCFeatureMode))

    return feature


# use multi processes to estimate the template length based on file size
# template_list is a list of template names
def generate_tplLengthfromList(template_list, tpl_root, cpu=20):
    # get template type
    file_type = get_tpl_type(tpl_root)
    pool = Pool(processes=cpu)
    tpl_length_list = []
    for i in range(len(template_list)):
        tplname = template_list[i]
        pool.apply_async(read_tplLength,
                         args=(tplname, tpl_root, file_type),
                         callback=tpl_length_list.append)
    pool.close()
    pool.join()
    return tpl_length_list


# read the template length directly
def read_tplLength(tplname, tpl_root, file_type):
    if not os.path.exists(
            os.path.join(tpl_root, "%s%s" % (tplname, file_type))):
        print("can not find %s%s in %s" % (tplname, file_type, tpl_root))
        sys.exit(-1)
    tpl = load_tpl(os.path.join(tpl_root, "%s%s" % (tplname, file_type)))
    return [tpl['name'], tpl['length']]


# get the template and tgt type
# only support .tpl and .tpl.pkl for template
# only support .tgt and .tgt.pkl for sequence
def get_tpltgt_type(tpl_root, tgt_root):
    tpl_type = get_tpl_type(tpl_root)
    tgt_type = get_tgt_type(tgt_root)
    return tpl_type, tgt_type


# get the template type
# only support .tpl and .tpl.pkl
def get_tpl_type(tpl_root):
    for filename0 in os.scandir(tpl_root):
        if filename0.name.endswith(".tpl"):
            file_type = ".tpl"
        elif filename0.name.endswith(".tpl.pkl"):
            file_type = ".tpl.pkl"
        else:
            print("we do not support this file_type in %s" % tpl_root)
            sys.exit(-1)
        break
    return file_type


# get the sequence type
# only support .tgt and .tgt.pkl
def get_tgt_type(tgt_root):
    for tgtname0 in os.scandir(tgt_root):
        if tgtname0.name.endswith(".tgt"):
            tgt_type = ".tgt"
        elif tgtname0.name.endswith(".tgt.pkl"):
            tgt_type = ".tgt.pkl"
        elif tgtname0.name.endswith(".hhm"):
            tgt_type = ".hhm"
        elif tgtname0.name.endswith(".hhm.pkl"):
            tgt_type = ".hhm.pkl"
        else:
            print("we do not support this file_type in %s" % tgt_root)
            sys.exit(-1)
        break
    return tgt_type


# get the distance type
# only support pairPotential.DFIRE16.pkl and predictedDistMatrix414C.pkl
def get_dist_type(dist_root):
    for distname in os.scandir(dist_root):
        if distname.name.endswith(".pairPotential.DFIRE16.pkl"):
            dist_type = ".pairPotential.DFIRE16.pkl"
        elif distname.name.endswith(".predictedDistMatrix414C.pkl"):
            dist_type = ".predictedDistMatrix414C.pkl"
        elif distname.name.endswith(".pairPotential.DFIRE.18.1.61.Wt4D.pkl"):
            dist_type = ".pairPotential.DFIRE.18.1.61.Wt4D.pkl"
        else:
            print("we do not support this file_type in %s" % dist_root)
            sys.exit(-1)
        break
    return dist_type
