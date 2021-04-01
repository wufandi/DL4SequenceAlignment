import torch
import os
import argparse
import pickle
import sys
import resource
import time
import datetime
import warnings
from torch.utils import data
from tqdm import tqdm
from torch.multiprocessing import Pool
from src.data.LoadDisPotential import Load_EdgeScore
from src.data.LoadTPLTGT import load_tgt, load_tpl
from src.data.LoadHHM import load_hhm
from src.data.DataProcessor import readDataList, get_tpl_type
from src.data.DataSet import BatchThreadingDataSet
from src.model.ObsModel import ObsModel
import src.model.Crf4SeqAlign as CrfModel
from src.model.BatchPair import Batchpair
from src.model.Utils import sortoutput, Compute_CbCb_distance_matrix
from src.model.Configure import ModelConfigure

resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_printoptions(edgeitems=1000)

example_text = "example:\n  OMP_NUM_THREADS=1 python3 NDThreaderSearch.py " \
        "-l example/T0954.template.txt " \
        "-m params/model.DA.SSA.1.pth params/model.DA.SSA.2.pth " \
        "-q example/T0954.tgt -d example/T0954.distPotential.DFIRE16.pkl " \
        "-t database/TPL_BC100 -o T0954"
parser = argparse.ArgumentParser(
        description="Thread a query sequence to a list of template protein "
        "with its distance potential, \nfind the best template and "
        "generate their alignment",
        usage="environment variable: set OMP_NUM_THREADS=1 \n"
        "python3 NDThreaderSearch.py -l TemplateList "
        "-m [Models] -t TPLs_path -q TGT -d DistancePotential "
        "[-a {Viterbi, MaxAcc}] [-k TopK] [-s sort_col] [-w Weight] "
        "[-i Iteration] [-g GPU] [-c CPU_num] "
        "[-o Output_path]",
        epilog=example_text,
        formatter_class=argparse.RawTextHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
optional = parser.add_argument_group("optional arguments")
other = parser.add_argument_group("other arguments")
required.add_argument(
        "-q", required=True,
        help="Query sequence in TGT format.")
required.add_argument(
        "-l", required=True,
        help="List of Template name for search.")
required.add_argument(
        "-m", required=True, default=[], nargs="+",
        help="Model path for test, can use more than one model\n"
        "More model you use, better threading result you get")
required.add_argument(
        "-d", required=True,
        help="Distance Potential for query sequence")
required.add_argument(
        "-t", required=True,
        help="Template proteins/TPLs path")
optional.add_argument(
        "-a", default="Viterbi", choices=["Viterbi", "MaxAcc"],
        help="Algorithm to initilize the alignment. [default = Viterbi]")
optional.add_argument(
        "-k", type=int, default=100,
        help="K best result of the threading output. (topK) "
        "[default = 100]")
optional.add_argument(
        "-s", type=int, default=12,
        help="Screen output is sorted with respect to "
        "specific column. [default=12 for objective score],\n"
        "only support 12 for main score, 13 for node score, "
        "14 for edge score, \n15 for norm (edge)score, "
        "16 for identical, 17 for id/ml")
optional.add_argument(
        "-i", type=int, default=10,
        help="Iteration number of ADMM algorithm, [default = 10]")
optional.add_argument(
        "-w", type=float,
        default=1, help="Weight of node score, "
        "which decides the objective function.\n"
        "the objective function = \n"
        "   node_weight * node_score + edge_score, [default = 1]")
optional.add_argument(
        "-g", type=int, nargs="+",
        default=[0],
        help="Gpu device you use to run your model. [default = 0]")
optional.add_argument(
        "-c", type=int,
        default=20,
        help="Cpu number you use in your project. [default = 20]")
optional.add_argument(
        "-o", default="",
        help="Output path [default = ${queryName}-${Template_list}]\n"
        "Its output consists of five parts:\n"
        "[1] all pairwise alignments, \n"
        "    save in output_path/alignments\n"
        "[2] rank file for each alignment in such format\n"
        "    1       2      3      4    5    6     7      8     "
        "  9     10     11      12        13      14        15      "
        " 16       17      18\n"
        "  q_name  t_name  q_len t_len col  q_gap t_gap  q_start "
        "q_end t_start t_end  main_sco node_sco  edge_sco norm_sco  "
        "seqid   id/ml  tra_num\n"
        "    save in output_path/${queryName}_list.Score\n"
        "[3] sorted rank file. Its sorted by main_sco (12nd) by default \n"
        "    save in output_path/${queryName}_list.SortedScore\n"
        "[4] topK sorted rank file. \n"
        "    save in output_path/${queryName}_list.SortedScore_topK\n"
        "[5] score: observation score matrix for each pair, "
        "you can use it directly without loading model\n"
        "    save in output_path/score")
other.add_argument(
        "--obs", default="",
        help="Observation score path, if already have the observation score "
        "for all pair,\ncan use them directly without loading model")
other.add_argument(
        "--onlyobs", default=0, choices=[0, 1], type=int,
        help="Only generate the DRNF observation score using gpu if you set"
        " it as 1.\n"
        "[default = 0]")
other.add_argument(
        "--extra",
        default=[], nargs="+",
        help="Extra initilized alignment path."
        "This alignment can be generated by other software,\n"
        "which can improve alignment quality \n"
        "You can use more than 1 other initilized alignment path")

args = parser.parse_args()

GPU = torch.device("cuda:%d" % sorted(args.g)[0] if
                   torch.cuda.is_available() else "cpu")
GPUsize = torch.cuda.get_device_properties(GPU).total_memory // (1024*1024)
MAXSIZE = 1000 * 800 * GPUsize // (12 * 1024)
CPU = torch.device('cpu')
numStates = 3


# main function to compute alignment
# 1. create a Batchpair class to store the information for two sequence
# 2. use viterbi algorithm to initialization the alignment
# 3. compute the dis_matrix for all templates
# 4. get the search space to speed up ADMM algorithm
# 5. use ADMM algorithm to get the alignment
# 6. compute the output score to rank all alignment
# return:
# tplnames(list of template name)
# output(list of score result)
# alignment_output(list of alignment file(.fasta) output)
def compute_alignment(tplname, observations, transitions,
                      pair_dis, disc_method, iteration,
                      edge_type="dist", Node_Weight=1):
    if args.q.endswith('.hhm') or args.q.endswith('.hhm.pkl'):
        tgt = load_hhm(args.q)
    else:
        tgt = load_tgt(args.q)
    tgtseq = tgt['sequence']
    tpl = load_tpl(os.path.join(args.t, tplname))
    tplseq = tpl['sequence']
    sequence = Batchpair(tpl['name'], tgt['name'],
                         tplseq, tgtseq, tpl, len(args.m))
    sequence.set_iter(iteration)
    # set the dis_matrix
    if 'atomDistMatrix' in tpl:
        dis_matrix = tpl['atomDistMatrix']['CbCb']
    else:
        dis_matrix = Compute_CbCb_distance_matrix(tpl)
    dis_matrix = torch.from_numpy(dis_matrix).float()
    dis_matrix = torch.where(torch.lt(dis_matrix, 0),
                             torch.ones(dis_matrix.size())*10000, dis_matrix)
    sequence.set_dismatrix(dis_matrix)
    observations = sequence.ModifyObs(observations, Node_Weight)
    alignment = sequence.alignment_init(
                observations, transitions, args.a)
    sequence.set_alignment(alignment)
    if args.extra != []:
        sequence.add_init_alignment(args.extra)
        observations = sequence.add_observation(observations)
    searchspace = sequence.template_search_space(disc_method, edge_type)
    sequence.set_searchspace(searchspace)
    alignment, output = sequence.ADMM_algorithm(
            observations, transitions, pair_dis,
            disc_method, edge_type, Node_Weight)
    sequence.set_output(output)
    alignment_output = sequence.get_alignment_output()
    return [tpl['name'], tgt['name'], alignment_output, output]


if __name__ == "__main__":
    # device
    if not torch.cuda.is_available():
        print("cuda is not avaiable")
        sys.exit(-1)
    # check the obs if it set
    if args.obs != '' and not os.path.exists(args.obs):
        print("the obs_root %s is not exists" % (args.obs))
        sys.exit(-1)
    if args.onlyobs == 1:
        ONLY_OBS_OPTION = True
    else:
        ONLY_OBS_OPTION = False
    #  load model
    obs_group = []
    crf_group = []
    SS3FeatureModes = []
    SS8FeatureModes = []
    ACCFeatureModes = []
    for model_name in args.m:
        if not os.path.exists(model_name):
            print("the model %s is not existing" % model_name)
            sys.exit(-1)
        model = torch.load(model_name, pickle_module=pickle,
                           map_location=lambda storage, loc: storage)
        modelConf = ModelConfigure(model['configure'])
        SS3FeatureModes.append(modelConf.SS3FeatureMode)
        SS8FeatureModes.append(modelConf.SS8FeatureMode)
        ACCFeatureModes.append(modelConf.ACCFeatureMode)

        print("Load %s.." % os.path.basename(model_name))
        modelConf.print_model_parameters()
        # obsmodel
        if args.obs == "":
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

    # check and load sequence(.tgt) file
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
        print("the query sequence is not existing")
        sys.exit(-1)
    # check extra path
    if args.extra != []:
        for path in args.extra:
            if not os.path.exists(path):
                print("the extra alignment path %s is not existing" % path)
                sys.exit(-1)
    # check and load pairwise potential (EPAD or dist_pot) file
    if not os.path.exists(args.d):
        print("the distance potential %s is not existing" % args.d)
        sys.exit(-1)
    # load distance potential
    pair_dis, disc_method, edge_type = Load_EdgeScore(args.d, tgt)
    pair_dis = torch.from_numpy(pair_dis)
    print("finish load %s" % args.d)
    pair_distance = pair_dis.detach().share_memory_()
    disc_method = torch.Tensor(disc_method).detach().share_memory_()

    # ADMM_ITERATION & node_weight & limit & sort_col
    if args.i < 0:
        print("ADMM_ITERATION %d should larger than 0" % args.i)
        sys.exit(-1)
    else:
        ADMM_ITERATION = args.i
    Node_Weight = args.w
    if args.w < 0.1:
        warnings.warn(
                "node_weight %.2f should larger than 0.1" % args.w, Warning)
    if args.s < 12 or args.s > 18:
        print("sort_col %d incorrect, must be integer from +12 to +18" %
              args.s)
        sys.exit(-1)
    sort_col = args.s
    # output root
    if args.o == "":
        output_root = tgt['name'] + '-' + os.path.basename(args.l)
    else:
        output_root = args.o

    # load pre-calculated observation
    if args.obs == '':
        obs_path = os.path.join(output_root, 'score')
    else:
        obs_path = args.obs
        print("read the observation score in %s" % obs_path)

    # create the dirs
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    if not os.path.exists(os.path.join(output_root, 'score')):
        os.makedirs(os.path.join(output_root, 'score'))

    if not os.path.exists(os.path.join(output_root, 'alignments')):
        os.makedirs(os.path.join(output_root, 'alignments'))

    print('start load template DataSet: %s..' % args.l)
    print('templates are in %s' % args.t)
    template_name_list = readDataList(args.l)
    datasize = len(template_name_list)
    tpl_type = get_tpl_type(args.t)
    if os.path.basename(args.q).endswith(".tgt"):
        tgt_type = ".tgt"
    elif os.path.basename(args.q).endswith(".tgt.pkl"):
        tgt_type = ".tgt.pkl"
    elif os.path.basename(args.q).endswith(".hhm"):
        tgt_type = ".hhm"
    elif os.path.basename(args.q).endswith(".hhm.pkl"):
        tgt_type = ".hhm.pkl"
    else:
        print("we only support tgt, tgt.pkl, hhm and hhm.pkl format "
              "for query sequence feature file")
        sys.exit(1)

    print('query sequence:                  %s' % args.q)
    print('query sequence name:             %s' % tgt['name'])
    print('query sequence length:           %s' % tgt['length'])
    print('distance potential:              %s' % args.d)
    print('list of template:                %s' % args.l)
    print('templates protein are in         %s' % args.t)
    print('output path:                     %s' % output_root)

    if args.obs == '':
        ThreadingSet = BatchThreadingDataSet(
                template_name_list, tgt, args.t,
                SS3FeatureModes, SS8FeatureModes, ACCFeatureModes, tpl_type,
                tpllimit=MAXSIZE//tgt['length'])
        print("Finish Template Sepreate and generate %d Template Group" %
              (len(ThreadingSet)))
        data_generator = data.DataLoader(
                ThreadingSet, batch_size=1, shuffle=False,
                num_workers=10, collate_fn=lambda x: x)
        print("start generating observation..")
        start = time.time()
        for number, pair_data in enumerate(tqdm(data_generator)):
            featdata, seqX, seqY, maskX, maskY = pair_data[0]
            batchSize, xLen, yLen, _ = featdata[0].size()

            observation = torch.zeros(len(args.m), batchSize, xLen, yLen, 3)
            with torch.no_grad():
                for j in range(len(args.m)):
                    observation[j] = obs_group[j](
                        featdata[j], seqX, seqY, maskX, maskY
                        )[0].detach().to(CPU)

            for ba in range(len(ThreadingSet.dataset[number])):
                tpl_length = maskX[ba]
                score = torch.zeros(len(args.m), tpl_length, yLen, 3)
                for j in range(len(args.m)):
                    score[j] = observation[j, ba, :tpl_length]
                tplname = ThreadingSet.dataset[number][ba]
                torch.save(score.half(), os.path.join(
                           output_root, "score", "%s-%s.pth" %
                           (tplname, tgt['name'])),
                           pickle_module=pickle)

        print('finish resnet obsScore calculation for %d data in %.2fs' %
              (datasize, (time.time()-start)))

    if ONLY_OBS_OPTION:
        sys.exit(-1)

    # empty gpu cache
    if args.obs == "":
        del observation, featdata, seqX, seqY, maskX, maskY
        del obsmodel, model0
        del data_generator, ThreadingSet
        with torch.cuda.device('cuda:%d' % sorted(args.g)[0]):
            torch.cuda.empty_cache()
    del obs_group, crf_group, crfmodel

    # compute the alignment
    print('start admm algorithm to get alignment..')
    print('algorithm of initialization:     %s' % args.a)
    print('weight of node score:            %d' % args.w)
    print('admm iteration:                  %d' % args.i)
    start = time.time()
    pbar3 = tqdm(range(datasize))
    pool = Pool(processes=args.c)
    alignment_result = []

    # get the output for alignment and update the pbar
    def getoutput(data):
        tplname = data[0]
        tgtname = data[1]
        alignment_output = data[2]
        score_output = data[3]
        alignment_result.append(
                [tplname, tgtname, alignment_output, score_output])
        with open(os.path.join(
                  output_root, "alignments", tplname + '-' +
                  tgtname + '.fasta'), 'w') as f:
            f.write(alignment_output)
        with open(os.path.join(
                  output_root, "%s_list.Score" % tgtname), "a") as F:
            F.write(score_output)
        pbar3.update()
        return

    for i in range(datasize):
        tplname = template_name_list[i]
        observations = torch.load(
                os.path.join(obs_path, '%s-%s.pth' % (tplname, tgt['name'])),
                pickle_module=pickle)
        observations = observations.float()
        observations = torch.mul(observations, Node_Weight)
        pool.apply_async(compute_alignment,
                         args=(tplname + tpl_type, observations,
                               transitions, pair_distance,
                               disc_method.tolist(),
                               ADMM_ITERATION, edge_type, Node_Weight),
                         callback=getoutput)
    pool.close()
    pool.join()
    pbar3.close()
    print('time: %.3f s' % (time.time()-start))
    print('finish admm algorithm')

    print('start generate output..')
    print('sort by index:                   %d' % sort_col)
    sortoutput(tgt['name'],
               os.path.join(output_root, tgt['name'] + '_list.Score'),
               args.k, sort_col)
    print("finish %d alignment generation and save them in %s" %
          (datasize, output_root))
    print("Date:                            %s" %
          (datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    print("Command:                         %s" % " ".join(sys.argv))
    print("Query name:                      %s" % tgt['name'])
    print("Template_list:                   %s" % os.path.basename(args.l))
    print("Output path:                     %s" % output_root)
