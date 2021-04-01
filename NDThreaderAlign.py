import torch
import os
import argparse
import pickle
import sys
import resource
import shutil
import time
import datetime
import warnings
from src.data.LoadDisPotential import Load_EdgeScore
from src.data.LoadTPLTGT import load_tgt, load_tpl
from src.data.LoadHHM import load_hhm
from src.data.DataProcessor import feature4sequence as featureCreate
from src.model.ObsModel import ObsModel
import src.model.Crf4SeqAlign as CrfModel
from src.model.Utils import Compute_CbCb_distance_matrix
from src.model.BatchPair import Batchpair
from src.model.Configure import ModelConfigure

resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_printoptions(edgeitems=1000)

example_text = "example:\n python3 NDThreaderAlign.py " \
        "-q example/T0954.tgt -t example/2gnqA.tpl.pkl" \
        "-d example/T0954.distPotential.DFIRE16.pkl" \
        "-m model/DRNFmodel.resnet50.DeepAlign.ss3_9_ss8_8_acc_9.pth " \
        "model/DRNFmodel.DeepAlign.ss3_0_ss8_0_acc_0.pth " \
        "-i 10 -w 1"
parser = argparse.ArgumentParser(
        description="Predict the alignment between a query sequence and a "
        "template protein with query sequence's distance potential"
        "and get its alignment score",
        usage="python3 NDThreaderAlign.py -q TGT -t TPL "
        "-d DistancePotential -m [Model] [-i Iteration] [-a {Viterbi, MaxAcc}]"
        " [-w Weight] [-g GPU] [-c CPU] [-o OUTPUT] ",
        epilog=example_text,
        formatter_class=argparse.RawTextHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
optional = parser.add_argument_group("optional arguments")
other = parser.add_argument_group("other arguments")
required.add_argument(
        "-m", required=True, default=[],
        nargs="+", help="Model path, can use more than one model\n"
        "More model you use, better alignment you get.")
required.add_argument(
        "-t", required=True,
        help="Template protein in TPL format.")
required.add_argument(
        "-q", required=True,
        help="Query sequence in TGT/HHM format.")
required.add_argument(
        "-d", required=True,
        help="Distance potential for query sequence.")
# default argument
optional.add_argument(
        "-a", default="Viterbi", choices=["Viterbi", "MaxAcc"],
        help="Algorithm to initilize the alignment. [default = Viterbi]")
optional.add_argument(
        "-i", type=int, default=4,
        help="Iteration number of ADMM algorithm. [default = 4]")
optional.add_argument(
        "-w", type=float,
        default=1, help="Weight of node score, "
        "which decides the objective function.\n"
        "the objective function = \n"
        "   node_weight * node_score + edge_score, [default = 1]")
optional.add_argument(
        "-o", default="",
        help="Output path, [default = $tplname-$queryname]\n"
        "Its output consists of two parts:\n"
        "[1] alignment file. Write pairwise alignment in FASTA format. \n"
        "    save in output_path/$tplname-$queryname.fasta\n"
        "[2] rank file for alignment in such format\n"
        "    1       2      3      4    5    6     7      8     "
        "  9     10     11      12        13      14        15      "
        " 16       17      18\n"
        "  q_name  t_name  q_len t_len col  q_gap t_gap  q_start "
        "q_end t_start t_end  main_sco node_sco  edge_sco norm_sco  "
        "seqid   id/ml  tra_num\n"
        "    save in output_path/${query_name}.Score\n"
        )
optional.add_argument(
        "-g", type=int, nargs="+",
        default=[0],
        help="Gpu device we use to run model, [default = 0]")
other.add_argument(
        "--print", type=int, default=0, choices=[0, 1],
        help="Print option. If the option is 1, it will print alignment score"
        "for each ADMM iteration.\n"
        "[default = 0]")

args = parser.parse_args()
GPU = torch.device("cuda:%d" % sorted(args.g)[0] if
                   torch.cuda.is_available() else "cpu")
CPU = torch.device('cpu')
numStates = 3

if __name__ == "__main__":
    # device
    if not torch.cuda.is_available():
        print("cuda is not avaiable")
        sys.exit(-1)
    if args.print == 1:
        PRINT_OPTION = True
    else:
        PRINT_OPTION = False
    # ----------------------- Load Model --------------------------- #
    start = time.time()
    #  model
    obs_group = []
    crf_group = []
    SS3FeatureModes = []
    SS8FeatureModes = []
    ACCFeatureModes = []
    for model_name in args.m:
        if not os.path.exists(model_name):
            print("the model %s is not exist" % model_name)
            sys.exit(-1)
        model = torch.load(model_name, pickle_module=pickle,
                           map_location=lambda storage, loc: storage)
        modelConf = ModelConfigure(model['configure'])
        SS3FeatureModes.append(modelConf.SS3FeatureMode)
        SS8FeatureModes.append(modelConf.SS8FeatureMode)
        ACCFeatureModes.append(modelConf.ACCFeatureMode)

        model0 = ObsModel(
            GPU, modelConf.feat1d, modelConf.feat2d,
            modelConf.layers1d, modelConf.neurons1d,
            modelConf.layers2d, modelConf.neurons2d, modelConf.dilation,
            modelConf.seqnet, modelConf.embedding, modelConf.pairwisenet,
            modelConf.block, modelConf.activation, modelConf.affine,
            modelConf.track_running_stats)

        obsmodel = torch.nn.DataParallel(model0, device_ids=sorted(args.g))
        obsmodel.module.load_state_dict(model['obsmodel'])
        obsmodel.to(GPU)
        obs_group.append(obsmodel)

        crfmodel = CrfModel.CRFLoss(numStates, CPU).to(CPU)
        crfmodel.load_state_dict(model['crfmodel'])
        crf_group.append(crfmodel)
    # get the transitions
    transitions = torch.zeros(len(args.m), 5, 5)
    for i in range(0, len(args.m)):
        transitions[i] = crf_group[i].trans + crf_group[i].P
    transitions = transitions * args.w
    transitions = transitions.to(CPU)
    transitions.detach_().share_memory_()
    print("finish load %d model in %.2f s.." %
          (len(args.m), time.time()-start))

    # ----------------------- Load data --------------------------- #
    start = time.time()
    # check and load template (.tpl) and sequence(.tgt) file
    if os.path.exists(args.t):
        tpl = load_tpl(args.t)
    else:
        print("the template is not exist")
        sys.exit(-1)
    if os.path.exists(args.q):
        if args.q.endswith('hhm') or args.q.endswith('.hhm.pkl'):
            if any(SS3FeatureModes) or any(SS8FeatureModes) \
                    or any(ACCFeatureModes):
                print("Please use TGT format file as input or use model "
                      "not using structure information")
                sys.exit(-1)
            tgt = load_hhm(args.q)
        else:
            tgt = load_tgt(args.q)
    else:
        print("the query sequence is not exist")
        sys.exit(-1)

    # check and load pairwise potential file
    if not os.path.exists(args.d):
        print("the distance potential %s is not exist" % args.d)
        sys.exit(-1)
    # load distance potential
    pair_dis, disc_method, edge_type = Load_EdgeScore(args.d, tgt)
    pair_dis = torch.from_numpy(pair_dis)
    pair_distance = pair_dis.detach().share_memory_()
    disc_method = torch.Tensor(disc_method).detach().share_memory_()
    print("finish load data in %.2f s.." % (time.time()-start))

    # ADMM_ITERATION & node_weight & limit
    if args.i < 0:
        print("ADMM_ITERATION %d should larger than 0" % args.i)
        sys.exit(-1)
    else:
        ADMM_ITERATION = args.i
    Node_Weight = args.w
    if args.w < 0.1:
        warnings.warn(
                "node_weight %.2f should larger than 0.1" % args.w, Warning)

    # output root
    if args.o == "":
        output_root = tpl['name'] + '-' + tgt['name']
    else:
        output_root = args.o

    # create the dirs
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    start = time.time()
    # create_feature
    xLen = tpl['length']
    yLen = tgt['length']
    seqX = torch.from_numpy(tpl['PSSM']).expand(1, xLen, 20)
    seqY = torch.from_numpy(tgt['PSSM']).expand(1, yLen, 20)
    maskX = torch.LongTensor([xLen])
    maskY = torch.LongTensor([yLen])

    featdata = []
    for ba in range(len(args.m)):
        feature = featureCreate(
            tpl, tgt, SS3FeatureModes[ba],
            SS8FeatureModes[ba], ACCFeatureModes[ba])
        featdata.append(feature)
    print("finish create feature in %.2f s.." % (time.time()-start))

    start = time.time()
    observations = torch.zeros(len(args.m), xLen, yLen, 3)

    with torch.no_grad():
        for j in range(len(args.m)):
            observations[j] = obs_group[j](
                featdata[j], seqX, seqY, maskX, maskY)[0].detach().to(CPU)
        observations = torch.mul(observations, Node_Weight)

    del obs_group, crf_group
    del obsmodel, model0, crfmodel
    with torch.cuda.device('cuda:%d' % sorted(args.g)[0]):
        torch.cuda.empty_cache()
    print("finish compute observations in %.2f s.." % (time.time()-start))

    sequence = Batchpair(tpl['name'], tgt['name'],
                         tpl['sequence'], tgt['sequence'],
                         tpl, len(args.m))
    sequence.set_iter(ADMM_ITERATION)

    # load dis_matrix
    print("load distance matrix from template file..")
    if 'atomDistMatrix' in tpl:
        dis_matrix = tpl['atomDistMatrix']['CbCb']
    else:
        dis_matrix = Compute_CbCb_distance_matrix(tpl)
    dis_matrix = torch.from_numpy(dis_matrix).float()
    dis_matrix = torch.where(
            torch.lt(dis_matrix, 0),
            torch.ones(dis_matrix.size())*10000, dis_matrix)
    sequence.set_dismatrix(dis_matrix)

    start = time.time()
    print("start running ADMM algorithm..")
    observations = sequence.ModifyObs(observations, Node_Weight)
    alignment = sequence.alignment_init(
            observations, transitions, args.a)
    sequence.set_alignment(alignment)
    searchspace = sequence.template_search_space(disc_method, edge_type)
    sequence.set_searchspace(searchspace)
    alignment, output = sequence.ADMM_algorithm(
            observations, transitions, pair_distance,
            disc_method.tolist(), edge_type, Node_Weight, Norm_Weight=5,
            PRINT_OPTION=PRINT_OPTION)
    sequence.set_output(output)
    alignment_output = sequence.get_alignment_output()
    print("finish ADMM algorithm in %.2f s.." % (time.time()-start))

    if os.path.exists(
            os.path.join(output_root,
                         tpl['name'] + '-' + tgt['name'] + '.Score')):
        os.remove(os.path.join(
                  output_root, tpl['name'] + '-' + tgt['name'] + '.Score'))
    with open(os.path.join(
              output_root, tgt['name'] + '.Score'), 'w') as score_output:
        score_output.write(output)
    print("result save in %s" % output_root)
    print("Query:               %s" % tgt['name'])
    print("Template:            %s" % tpl['name'])
    print("Distance potential:  %s" % os.path.basename(args.d))
    print("Model:               %s" % "\n                     ".join(
          [os.path.basename(modelname) for modelname in args.m]))
    print("Query Length:        %d" % tgt['length'])
    print("Template Length:     %d" % tpl['length'])
    print("Main score:          %.2f" % sequence.maxscore)
    print("Node score:          %.2f" % sequence.nodescore)
    print("Edge score:          %.2f" % sequence.edgescore)
    print("Norm score:          %.2f" % sequence.normscore)
    print("Date:                %s" %
          (datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    print("Command:         %s" % " ".join(sys.argv))
    print("\nalignment:")
    print(alignment_output)
    with open(os.path.join(
              output_root,
              tpl['name'] + '-' + tgt['name'] + '.fasta'), 'w') as f:
        f.write(alignment_output)
    print("alignment_score:")
    print("%s" % output)
