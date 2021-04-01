import torch
import tqdm
import os
import pickle
import sys
import argparse
import time
import datetime
from torch.utils import data
from torch.multiprocessing import Pool
from src.data.LoadTPLTGT import load_tgt, load_tpl
from src.data.LoadHHM import load_hhm
from src.data.DataProcessor import readDataList, get_tpl_type
from src.model.ObsModel import ObsModel
from src.model.BatchPair import Batchpair
from src.data.DataSet import BatchThreadingDataSet
from src.model import Crf4SeqAlign as CrfModel
from src.model.Utils import sortoutput
from src.model.Configure import ModelConfigure

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_printoptions(edgeitems=100)

example_text = "example:\n   OMP_NUM_THREADS=1 python3 DRNFSearch.py " \
        "-m params/model.DA.SSA.1.pth " \
        "-l example/T0954.template.txt " \
        "-t database/TPL_BC100 -q example/T0954.tgt " \
        "-o T0954"
parser = argparse.ArgumentParser(
        description="Search a best template protein from a template list "
        "for a query sequence and generate their alignment",
        usage="environment variable: set OMP_NUM_THREADS=1 \n"
        "python3 DRNFSearch.py -l TemplateList "
        "-m [Models] -q TGT -t TPLs_path "
        "[-a {Viterbi, MaxAcc}] "
        "[-k TopK] [-g GPU] [-c CPU_num] "
        "[-o Output_path]",
        epilog=example_text,
        formatter_class=argparse.RawTextHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
optional = parser.add_argument_group("optional arguments")
other = parser.add_argument_group("other arguments")
required.add_argument(
        "-q", required=True,
        help="Query sequence in TGT/HMM format.")
required.add_argument(
        "-l", required=True,
        help="List of Template name for search.")
required.add_argument(
        "-m", required=True, default=[], nargs="+",
        help="Model path for test, can use more than one model\n"
        "More model you use, better alignment you get")
required.add_argument(
        "-t", required=True,
        help="Template proteins/TPLs path")
optional.add_argument(
        "-a", default="Viterbi", choices=["Viterbi", "MaxAcc", "Mix"],
        help="Algorithm use to generate alignment from score matrix,\n"
        " only support Viterbi and MaxAcc, [default = Viterbi]")
optional.add_argument(
        "-k", type=int, default=100,
        help="K best result of the threading output. (topK) "
        "[default = 100]")
optional.add_argument(
        "-g", type=int, nargs="+", default=[0],
        help="Gpu device you use to run our model [default = 0]")
optional.add_argument(
        "-c", type=int, default=10,
        help="Cpu number you use in your project [default = 10]")
optional.add_argument(
        "-o", default="",
        help="Output path [default = ${queryName}-align]\n"
        "Its output consists of five parts:\n"
        "[1] all pairwise alignments, \n"
        "    save in output_path/alignments\n"
        "[2] rank file for each alignment in such format\n"
        "    1       2      3      4    5    6     7      8     "
        "  9     10      11      12         13      14\n"
        "  q_name  t_name  q_len t_len col  q_gap t_gap  q_start "
        "q_end t_start t_end viterbi_sco  seqid   id/ml\n"
        "    save in output_path/${queryName}_list.Score\n"
        "[3] sorted rank file. It sort by each viterbi_sco (12nd) \n"
        "    save in output_path/${queryName}_list.SortedScore\n"
        "[4] topK sorted rank file. \n"
        "    save in output_path/${queryName}_list.SortedScore_topK\n"
        "[5] score: observation score matrix for each pair, "
        "you can use it directly without loading model\n"
        "    save in output_path/score")
other.add_argument(
        "--obs", default="",
        help="Observation score path, if already have the observation score "
        "for all pair,\ncan use them directly without loading model"
        " / using GPU")
other.add_argument(
        "--onlyobs", default=0, choices=[0, 1], type=int,
        help="Only generate the DRNF observation score using gpu if you set"
        " it as 1.\n"
        "[default = 0]")
args = parser.parse_args()


# ARGS
GPU = torch.device("cuda:%d" % sorted(args.g)[0] if
                   torch.cuda.is_available() else "cpu")
GPUsize = torch.cuda.get_device_properties(GPU).total_memory // (1024*1024)
MAXSIZE = 800 * 1000 * GPUsize // (12 * 1024)
CPU = torch.device("cpu")
numStates = 3
workdir = os.getcwd()


def generateAlign(tplname, tgtname, observations, transitions):
    tpl = load_tpl(os.path.join(args.t, tplname))
    if args.q.endswith('.hhm') or args.q.endswith('.hhm.pkl'):
        tgt = load_hhm(args.q)
    else:
        tgt = load_tgt(args.q)
    tgtseq = tgt['sequence']
    tplseq = tpl['sequence']
    sequence = Batchpair(tpl['name'], tgt['name'], tplseq, tgtseq, tpl)
    observation = torch.mean(observations, 0, keepdim=True)
    transition = torch.mean(transitions, 0, keepdim=True)
    alignments = sequence.alignment_init(observation, transition, args.a)
    output = sequence.get_CNF_output(alignments[0], observation,
                                     transition, args.a)
    sequence.maxalign = alignments[0]
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
    # MODEL
    obs_group = []
    crf_group = []
    SS3FeatureModes = []
    SS8FeatureModes = []
    ACCFeatureModes = []
    for model_name in args.m:
        if not os.path.exists(model_name):
            print("model %s is not existing" % model_name)
            sys.exit(-1)
        model = torch.load(model_name, pickle_module=pickle,
                           map_location=lambda storage, loc: storage)
        modelConf = ModelConfigure(model['configure'])
        SS3FeatureModes.append(modelConf.SS3FeatureMode)
        SS8FeatureModes.append(modelConf.SS8FeatureMode)
        ACCFeatureModes.append(modelConf.ACCFeatureMode)

        print("Load %s.." % os.path.basename(model_name))
        modelConf.print_model_parameters()
        if args.obs == "":
            model0 = ObsModel(
                GPU, modelConf.feat1d, modelConf.feat2d,
                modelConf.layers1d, modelConf.neurons1d,
                modelConf.layers2d, modelConf.neurons2d, modelConf.dilation,
                modelConf.seqnet, modelConf.embedding, modelConf.pairwisenet,
                modelConf.block, modelConf.activation,
                modelConf.affine, modelConf.track_running_stats)

            obsmodel = torch.nn.DataParallel(model0, device_ids=sorted(args.g))
            obsmodel.module.load_state_dict(model['obsmodel'])
            obsmodel.to(GPU)
            obs_group.append(obsmodel)

        crfmodel = CrfModel.CRFLoss(numStates, CPU).to(CPU)
        crfmodel.load_state_dict(model['crfmodel'])
        crf_group.append(crfmodel)
    print("finish loading %d model" % len(args.m))

    # get the transitions
    transitions = torch.zeros(len(args.m), 5, 5)
    for i in range(0, len(args.m)):
        transitions[i] = crf_group[i].trans + crf_group[i].P
    transitions = transitions.to(CPU)
    transitions.detach_().share_memory_()

    # testlist
    if not os.path.exists(args.l):
        print("input tpllist path %s is not existing" % args.l)
        sys.exit(-1)
    # tgt
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
    # output
    if args.o == "":
        outputname = tgt['name'] + '-align'
    else:
        outputname = args.o
    if not os.path.exists(outputname):
        os.makedirs(outputname)
    if not os.path.exists(os.path.join(outputname, "score")):
        os.makedirs(os.path.join(outputname, "score"))

    if not os.path.exists(os.path.join(outputname, 'alignments')):
        os.makedirs(os.path.join(outputname, 'alignments'))
    if args.obs == "":
        obs_path = os.path.join(outputname, 'score')
    else:
        obs_path = args.obs
        print("read the Observation score in %s" % obs_path)

    # load template list
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
    print('list of template:                %s' % args.l)
    print('templates protein are in         %s' % args.t)
    print('output path:                     %s' % outputname)

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
        print("start generating observation score..")
        start = time.time()
        for number, pair_data in enumerate(tqdm.tqdm(data_generator)):
            featdata, seqX, seqY, maskX, maskY = pair_data[0]
            batchSize, xLen, yLen, _ = featdata[0].size()

            observation = torch.zeros(len(args.m), batchSize, xLen, yLen, 3)
            with torch.no_grad():
                for j in range(len(args.m)):
                    observation[j] = obs_group[j](
                        featdata[j], seqX, seqY, maskX, maskY
                        ).detach().to(CPU)

            for ba in range(len(ThreadingSet.dataset[number])):
                tpl_length = maskX[ba]
                score = torch.zeros(len(args.m), tpl_length, yLen, 3)
                for j in range(len(args.m)):
                    score[j] = observation[j, ba, :tpl_length]
                tplname = ThreadingSet.dataset[number][ba]
                torch.save(score.half(), os.path.join(
                           outputname, "score", "%s-%s.pth" %
                           (tplname, tgt['name'])),
                           pickle_module=pickle)

        print("finish %d observations generation in %.2fs" %
              (datasize, time.time()-start))

    if ONLY_OBS_OPTION:
        sys.exit(-1)

    # empty the cache
    if args.obs == "":
        del observation, featdata, seqX, seqY, maskX, maskY
        del obsmodel, model0
        del data_generator, ThreadingSet
        with torch.cuda.device(GPU):
            torch.cuda.empty_cache()
    del obs_group, crf_group, crfmodel

    print("start calculating alignment..")
    print('algorithm of initialization:     %s' % args.a)
    start = time.time()
    pbar = tqdm.tqdm(total=datasize)
    pool = Pool(processes=args.c)

    def getoutput(data):
        tplname = data[0]
        tgtname = data[1]
        alignment_output = data[2]
        score_output = data[3]
        with open(os.path.join(
                  outputname, "alignments", tplname + '-' +
                  tgtname + '.fasta'), 'w') as f:
            f.write(alignment_output)
        with open(os.path.join(
                  outputname, "%s_list.Score" % tgt['name']), "a") as F:
            F.write(score_output)
        pbar.update()
        return

    for i in range(datasize):
        tplname = template_name_list[i]
        observations = torch.load(
                os.path.join(obs_path, "%s-%s.pth" %
                             (tplname, tgt['name'])),
                pickle_module=pickle)
        observations = observations.float()
        pool.apply_async(generateAlign,
                         args=(tplname + tpl_type, tgt['name'] + tgt_type,
                               observations, transitions),
                         callback=getoutput)
    pool.close()
    pool.join()
    pbar.close()
    print("finish calculating alignment in %.2fs" % (time.time()-start))
    sortoutput(tgt['name'],
               os.path.join(outputname, tgt['name'] + '_list.Score'),
               args.k)
    print("finish %d alignment generation and save them in %s" %
          (datasize, outputname))
    print("Date:                            %s" %
          (datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    print("Command:                         %s" % " ".join(sys.argv))
    print("Query name:                      %s" % tgt['name'])
    print("Template_list:                   %s" % os.path.basename(args.l))
    print("Output path:                     %s" % outputname)
