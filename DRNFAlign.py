import torch
import argparse
import pickle
import sys
import os
import time
import datetime
import numpy as np
from src.data.LoadTPLTGT import load_tpl, load_tgt
from src.data.LoadHHM import load_hhm
from src.data.DataProcessor import feature4sequence
from src.model.ObsModel import ObsModel
from src.model import Crf4SeqAlign as CrfModel
from src.model.Utils import alignment_output, getStateNum
from src.model.Configure import ModelConfigure


example_text = "example:\n python3 DRNFAlign.py " \
        "-m params/model.DA.SSA.1.pth " \
        "-t example/2gnqA.tpl.pkl " \
        "-q example/T0954.tgt"
parser = argparse.ArgumentParser(
        description="Predict the alignment between a query sequence/TGT/HHM"
        "and a template protein/TPL",
        usage="python3 DRNFAlign.py -m Model -t TPL -q TGT "
        "[-a {Viterbi, MaxAcc}] [-g GPU] [-o OUTPUT]",
        epilog=example_text,
        formatter_class=argparse.RawTextHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
optional = parser.add_argument_group("optional arguments")
required.add_argument(
        "-m", required=True,
        help="Model path")
required.add_argument(
        "-t", required=True,
        help="Template protein in TPL format.")
required.add_argument(
        "-q", required=True,
        help="Query sequence in TGT/HHM format.")
optional.add_argument(
        "-a", choices=["Viterbi", "MaxAcc"],
        default="Viterbi",
        help="Algorithm use to generate alignment from score matrix, \n"
        "only support Viterbi and MaxAcc, [default = Viterbi]")
optional.add_argument(
        "-g", type=int, default=0,
        help="Gpu device we use to run model [default = 0]")
optional.add_argument(
        "-o", default="",
        help="Output file. Write pairwise alignment in FASTA format. \n"
        "[default = template_name-sequence_name.fasta]")
args = parser.parse_args()

GPU = torch.device("cuda:%d" % args.g)
CPU = torch.device("cpu")

torch.set_printoptions(edgeitems=20)
numStates = 3

if __name__ == "__main__":
    # device
    if not torch.cuda.is_available():
        print("cuda is not avaiable")
        sys.exit(-1)
    # model
    if not os.path.exists(args.m):
        print("model %s is not existing" % args.m)
        sys.exit(-1)
    model = torch.load(args.m, pickle_module=pickle,
                       map_location=lambda storage, loc: storage)
    modelConf = ModelConfigure(model['configure'])
    SS3FeatureMode = modelConf.SS3FeatureMode
    SS8FeatureMode = modelConf.SS8FeatureMode
    ACCFeatureMode = modelConf.ACCFeatureMode
    modelConf.print_model_parameters()
    obsmodel = ObsModel(
        GPU, modelConf.feat1d, modelConf.feat2d,
        modelConf.layers1d, modelConf.neurons1d,
        modelConf.layers2d, modelConf.neurons2d, modelConf.dilation,
        modelConf.seqnet, modelConf.embedding, modelConf.pairwisenet,
        modelConf.block, modelConf.activation, modelConf.affine,
        modelConf.track_running_stats)
    obsmodel.load_state_dict(model['obsmodel'])
    obsmodel.to(GPU)
    crfmodel = CrfModel.CRFLoss(numStates, CPU).to(CPU)
    crfmodel.load_state_dict(model['crfmodel'])

    # should read the Template and Sequence
    if os.path.exists(args.t):
        tpl = load_tpl(args.t)
    else:
        print('the tpl file is not existing')
        sys.exit()

    if os.path.exists(args.q):
        if args.q.endswith('hhm') or args.q.endswith('.hhm.pkl'):
            if SS3FeatureMode != 0 or SS8FeatureMode != 0 \
                    or ACCFeatureMode != 0:
                print("Please use TGT format file as input or use model "
                      "not using structure information")
                sys.exit(-1)
            tgt = load_hhm(args.q)
        else:
            tgt = load_tgt(args.q)
    else:
        print('the tgt file is not existing')
        sys.exit()

    tpl_name = tpl['name']
    tgt_name = tgt['name']

    # output name
    if args.o == "":
        outputname = "%s-%s.fasta" % (tpl_name, tgt_name)
    else:
        outputname = args.o
    xLen = tpl['length']
    yLen = tgt['length']

    feature = feature4sequence(
            tpl, tgt, SS3FeatureMode, SS8FeatureMode, ACCFeatureMode).to(GPU)
    seqX = torch.from_numpy(tpl['PSSM']).expand(1, xLen, 20).to(GPU)
    seqY = torch.from_numpy(tgt['PSSM']).expand(1, yLen, 20).to(GPU)
    maskX = torch.LongTensor([xLen])
    maskY = torch.LongTensor([yLen])

    model_start = time.time()
    with torch.no_grad():
        observation = obsmodel(feature, seqX, seqY,
                               maskX, maskY).detach().to(CPU)
    trans = crfmodel.trans + crfmodel.P
    trans = trans.expand(1, 5, 5)

    if args.a == "Viterbi":
        testScore, testAlignment = CrfModel.viterbi(
                observation, trans, tpl, tgt, MODIFY=False)
        rankscore = CrfModel.get_CNF_output(tpl, tgt, testAlignment[0],
                                            observation, trans, args.a)
    elif args.a == "MaxAcc":
        mask = torch.LongTensor([[xLen, yLen]])
        mas = np.array(mask).astype(np.int32)
        obs = observation.detach().numpy()
        tra = trans.detach().numpy()
        partition1, alpha = CrfModel._crf_BatchForward(obs, trans, mask)
        partition2, beta = CrfModel._crf_BatchBackward(obs, trans, mask)
        partition = (partition1 + partition2) / 2
        singleton_score = CrfModel.marginalMatrix(
                observation, alpha, beta, partition, mask)
        testScore, testAlignment = CrfModel.BatchViterbi4alignment(
                singleton_score, tpl, tgt)
        rankscore = CrfModel.get_CNF_output(tpl, tgt, testAlignment[0],
                                            singleton_score, trans, args.a)
    match_col, _, _ = getStateNum(testAlignment[0])
    print("Query:           %s" % tgt['name'])
    print("Template:        %s" % tpl['name'])
    print("Model:           %s" % os.path.basename(args.m))
    print("Query Length:    %d" % tgt['length'])
    print("Template Length: %d" % tpl['length'])
    print("Time:            %.3f" % (time.time()-model_start))
    print("Match_columns:   %d" % match_col)
    if args.a == "Viterbi":
        print("Viterbi Score:   %.2f" % (testScore.item()))
        print("Eval Score:      %.2f" %
              (testScore.item() / min(xLen, yLen) * 100))
    elif args.a == "MaxAcc":
        print("MaxAcc Score:   %.2f" % (testScore.item()))
    print("Date:            %s" %
          (datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    print("Command:         %s" % " ".join(sys.argv))
    print("Output:          %s\n" % outputname)
    print("%s\n" % rankscore)

    output = alignment_output(
            tpl_name, tgt_name, tpl['sequence'],
            tgt['sequence'], testAlignment[0])
    with open(outputname, "w") as F:
        F.write(output)
    print(output)
