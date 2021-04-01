import os
import torch
from torch.utils.data import Dataset
from .ReadAlignments import ReadAlignments, refactor_Single_alignment
from .LoadTPLTGT import load_tpl, load_tgt
from .LoadHHM import load_hhm
from .DataProcessor import feature4sequence, TemplateSepreate
from .FeatureCreate import alignmentMatrix, distanceFeature
from .LoadDisPotential import Load_EdgeScore


class AlignmentDataSet(Dataset):
    # alignment dataset for pytorch
    # It's used for training model and test loss
    # dataset is a list of alignment list
    # [[tplname1, tgtname1], [tplname2, tgtname2], ...]
    def __init__(self, dataset, tpl_root, tgt_root, alignment_root,
                 SS3FeatureMode, SS8FeatureMode, ACCFeatureMode,
                 tpl_type='.tpl', tgt_type='.tgt'):
        # Initialization
        self.dataset = dataset
        self.tpl_root = tpl_root
        self.tgt_root = tgt_root
        self.alignment_root = alignment_root
        self.tpl_type = tpl_type
        self.tgt_type = tgt_type
        self.SS3FeatureMode = SS3FeatureMode
        self.SS8FeatureMode = SS8FeatureMode
        self.ACCFeatureMode = ACCFeatureMode

    def __len__(self):
        # Denotes the total number of samples
        return len(self.dataset)

    def __getitem__(self, index):
        # Generate one sample of data
        tpl_data = []
        tgt_data = []
        batchSize = len(self.dataset[index])
        alignments_path = []
        alignments_names = []
        for ba in range(batchSize):
            if len(self.dataset[index][ba].split('-')) == 3:
                tplname, tgtname, domainID = self.dataset[index][ba].split('-')
                tgtname = "%s-%s" % (tgtname, domainID)
            else:
                tplname, tgtname = self.dataset[index][ba].split('-')
            tpl = load_tpl(os.path.join(
                           self.tpl_root, tplname + self.tpl_type))
            tpl_data.append(tpl)
            if self.tgt_type == '.hhm' or self.tgt_type == '.hhm.pkl':
                tgt = load_hhm(os.path.join(
                               self.tgt_root, tgtname + self.tgt_type))
            else:
                tgt = load_tgt(os.path.join(
                               self.tgt_root, tgtname + self.tgt_type))
            tgt_data.append(tgt)
            alignments_path.append(
                    os.path.join(self.alignment_root, '%s-%s.fasta' %
                                 (tplname, tgtname)))
            alignments_names.append([tplname, tgtname])

        maskX = torch.LongTensor([TPL['length'] for TPL in tpl_data])
        maskY = torch.LongTensor([TGT['length'] for TGT in tgt_data])
        xLen = torch.max(maskX).item()
        yLen = torch.max(maskY).item()

        featsize = 10 + self.SS3FeatureMode + \
            self.SS8FeatureMode + self.ACCFeatureMode
        featdata = torch.zeros(batchSize, xLen, yLen, featsize)
        seqX = torch.zeros(batchSize, xLen, 20)
        seqY = torch.zeros(batchSize, yLen, 20)

        for ba in range(batchSize):
            featdata[ba, :maskX[ba], :maskY[ba], :] = feature4sequence(
                    tpl_data[ba], tgt_data[ba], self.SS3FeatureMode,
                    self.SS8FeatureMode, self.ACCFeatureMode)
            seqX[ba, :maskX[ba]] = torch.from_numpy(tpl_data[ba]['PSSM'])
            seqY[ba, :maskY[ba]] = torch.from_numpy(tgt_data[ba]['PSSM'])

        alignments = ReadAlignments(alignments_path, alignments_names)
        return alignments, featdata, seqX, seqY, maskX, maskY


class NDTAlignmentDataSet(Dataset):
    # Generate Alignment and Threading result with singleton feature,
    # pairwise information feature and initial alignment
    def __init__(self, dataset, tpl_root, tgt_root,
                 singleton_root, pairwise_root, init_root, method,
                 DistFeatureMode, NormFeatureMode, AlignFeatureMode,
                 tpl_type='.tpl', tgt_type='.tgt', singleton_type='.DRNF.pkl',
                 pairwise_type='.pairPotential.DFIRE16.pkl'):
        # Initialization
        self.dataset = dataset
        self.tpl_root = tpl_root
        self.tgt_root = tgt_root
        self.singleton_root = singleton_root
        self.pairwise_root = pairwise_root
        self.init_root = init_root
        self.method = method
        self.tpl_type = tpl_type
        self.tgt_type = tgt_type
        self.DistFeatureMode = DistFeatureMode
        self.NormFeatureMode = NormFeatureMode
        self.AlignFeatureMode = AlignFeatureMode
        self.pairwise_type = pairwise_type
        self.singleton_type = singleton_type

    def __len__(self):
        # Denotes the total number of samples
        return len(self.dataset)

    def __getitem__(self, index):
        if len(self.dataset[index].split('-')) == 3:
            tgtname, domainID, tplname = self.dataset[index].split('-')
            tgtname = "%s-%s" % (tgtname, domainID)
        else:
            tgtname, tplname = self.dataset[index].split('-')
        tpl = load_tpl(os.path.join(self.tpl_root, tplname + self.tpl_type))
        if self.tgt_type == '.hhm' or self.tgt_type == '.hhm.pkl':
            tgt = load_hhm(
                    os.path.join(self.tgt_root, tgtname + self.tgt_type))
        else:
            tgt = load_tgt(
                    os.path.join(self.tgt_root, tgtname + self.tgt_type))

        xLen = tpl['length']
        yLen = tgt['length']
        maskX = torch.LongTensor([xLen])
        maskY = torch.LongTensor([yLen])

        observation = torch.load(
            os.path.join(self.singleton_root, "%s-%s%s" %
                         (tplname, tgtname, self.singleton_type)))
        modelSize = observation.size(0)
        init_alignment = []
        alignment_paths = []
        for ba in range(modelSize):
            if os.path.exists(
                    os.path.join(self.init_root, '%s-%s.%d.fasta' %
                                 (tplname, tgtname, ba))):
                alignment_path = os.path.join(
                    self.init_root, '%s-%s.%d.fasta' % (tplname, tgtname, ba))
            else:
                alignment_path = os.path.join(
                    self.init_root, '%s-%s.fasta' % (tplname, tgtname))

            alignment_paths.append(alignment_path)

        init_alignment = ReadAlignments(
            alignment_paths, [[tplname, tgtname]] * modelSize)

        dis_matrix = torch.from_numpy(tpl['atomDistMatrix']['CbCb']).float()
        tgt_dis, tgt_disc, _ = Load_EdgeScore(
            os.path.join(self.pairwise_root, tgtname + self.pairwise_type),
            tgt)
        tgt_dis = torch.from_numpy(tgt_dis)

        obs_data = []
        dist_data = []
        Distance_feat = torch.zeros(
            modelSize, xLen, yLen,
            self.DistFeatureMode+self.NormFeatureMode+self.AlignFeatureMode)

        for ba in range(modelSize):
            index = 0
            alignment = refactor_Single_alignment(init_alignment[ba])
            Distance_feat[ba, :, :, index:index+self.AlignFeatureMode] = \
                alignmentMatrix(alignment, xLen, yLen, self.AlignFeatureMode)
            index += self.AlignFeatureMode

            dist_feat, norm_feat = distanceFeature(
                alignment, tgt_dis, dis_matrix, tgt_disc,
                self.DistFeatureMode, self.NormFeatureMode)
            Distance_feat[ba, :, :, index:index+self.DistFeatureMode] = \
                dist_feat
            index += self.DistFeatureMode

            Distance_feat[ba, :, :, index:index+self.NormFeatureMode] = \
                norm_feat
            obs_data.append(observation[ba].unsqueeze(0))
            dist_data.append(Distance_feat[ba].unsqueeze(0))

        return obs_data, dist_data, maskX, maskY


class PairwiseDataSet(Dataset):
    # pairwise dataset for pytorch
    # it's used for batch alignment / threading project
    # dataset is a list of pairwise name
    def __init__(self, dataset, tpl_root, tgt_root,
                 SS3FeatureModes, SS8FeatureModes, ACCFeatureModes,
                 tpl_type='.tpl', tgt_type='.tgt'):
        self.dataset = dataset
        self.tgt_root = tgt_root
        self.tpl_root = tpl_root
        self.SS3FeatureModes = SS3FeatureModes
        self.SS8FeatureModes = SS8FeatureModes
        self.ACCFeatureModes = ACCFeatureModes
        self.ModelSize = len(SS3FeatureModes)
        self.tpl_type = tpl_type
        self.tgt_type = tgt_type

    def __len__(self):
        # Denotes the total number of samples
        return len(self.dataset)

    def __getitem__(self, index):
        # Generate one sample of data
        if len(self.dataset[index].split('-')) == 3:
            tgtname, domainID, tplname = self.dataset[index].split('-')
            tgtname = "%s-%s" % (tgtname, domainID)
        else:
            tgtname, tplname = self.dataset[index].split('-')
        tpl = load_tpl(os.path.join(self.tpl_root, tplname + self.tpl_type))
        if self.tgt_type == '.hhm' or self.tgt_type == '.hhm.pkl':
            tgt = load_hhm(
                    os.path.join(self.tgt_root, tgtname + self.tgt_type))
        else:
            tgt = load_tgt(
                    os.path.join(self.tgt_root, tgtname + self.tgt_type))
        xLen = tpl['length']
        yLen = tgt['length']
        maskX = torch.LongTensor([xLen])
        maskY = torch.LongTensor([yLen])

        seqX = torch.from_numpy(tpl['PSSM']).expand(1, xLen, 20)
        seqY = torch.from_numpy(tgt['PSSM']).expand(1, yLen, 20)
        featdata = []

        for num in range(self.ModelSize):
            featsize = 10 + self.SS3FeatureModes[num] + \
                self.SS8FeatureModes[num] + self.ACCFeatureModes[num]
            feature = torch.zeros(1, xLen, yLen, featsize)
            feature[0, :, :, :] = feature4sequence(
                        tpl, tgt, self.SS3FeatureModes[num],
                        self.SS8FeatureModes[num], self.ACCFeatureModes[num])
            featdata.append(feature)

        return featdata, seqX, seqY, maskX, maskY


class ThreadingDataSet(Dataset):
    # threading dataset for pytorch
    # it's used for threading search project
    # search a template set for specific target
    # dataset is a list of template name
    def __init__(self, dataset, tgt, tpl_root,
                 SS3FeatureModes, SS8FeatureModes, ACCFeatureModes,
                 tpl_type='.tpl'):
        self.dataset = dataset
        self.tgt = tgt
        self.tpl_root = tpl_root
        self.SS3FeatureModes = SS3FeatureModes
        self.SS8FeatureModes = SS8FeatureModes
        self.ACCFeatureModes = ACCFeatureModes
        self.ModelSize = len(SS3FeatureModes)
        self.tpl_type = tpl_type

    def __len__(self):
        # Denotes the total number of samples
        return len(self.dataset)

    def __getitem__(self, index):
        # Generate one sample of data
        tplname = self.dataset[index]
        tpl = load_tpl(os.path.join(self.tpl_root, tplname + self.tpl_type))
        xLen = tpl['length']
        yLen = self.tgt['length']
        maskX = torch.LongTensor([xLen])
        maskY = torch.LongTensor([yLen])

        seqX = torch.from_numpy(tpl['PSSM']).expand(1, xLen, 20)
        seqY = torch.from_numpy(self.tgt['PSSM']).expand(1, yLen, 20)
        featdata = []

        for num in range(self.ModelSize):
            featsize = 10 + self.SS3FeatureModes[num] + \
                self.SS8FeatureModes[num] + self.ACCFeatureModes[num]
            feature = torch.zeros(1, xLen, yLen, featsize)
            feature[0, :, :, :] = feature4sequence(
                        tpl, self.tgt, self.SS3FeatureModes[num],
                        self.SS8FeatureModes[num], self.ACCFeatureModes[num])
            featdata.append(feature)

        return featdata, seqX, seqY, maskX, maskY


class BatchThreadingDataSet(Dataset):
    # threading dataset for pytorch
    # to speed up GPU time for ResNet model
    # First, reading length of all template Length and them put all template
    #   to different group
    def __init__(self, dataset, tgt, tpl_root,
                 SS3FeatureModes, SS8FeatureModes, ACCFeatureModes,
                 tpl_type='.tpl', tpllimit=1000):
        self.tgt = tgt
        self.tpl_root = tpl_root
        self.SS3FeatureModes = SS3FeatureModes
        self.SS8FeatureModes = SS8FeatureModes
        self.ACCFeatureModes = ACCFeatureModes
        self.ModelSize = len(SS3FeatureModes)
        self.tpl_type = tpl_type
        self.dataset = TemplateSepreate(dataset, tpl_root, tpllimit)

    def __len__(self):
        # Denotes the total number of samples
        return len(self.dataset)

    def __getitem__(self, index):
        # Generate one sample of data
        tpl_names = self.dataset[index]
        batchSize = len(tpl_names)
        tpl_group = []
        featdata = []
        for tplname in tpl_names:
            tpl_group.append(
                load_tpl(os.path.join(self.tpl_root, tplname+self.tpl_type)))

        xLen = max([tpl['length'] for tpl in tpl_group])
        yLen = self.tgt['length']
        maskX = torch.LongTensor([tpl['length'] for tpl in tpl_group])
        maskY = torch.LongTensor([yLen] * batchSize)

        seqX = torch.zeros(batchSize, xLen, 20)
        seqY = torch.from_numpy(self.tgt['PSSM']).expand(batchSize, yLen, 20)

        for ba in range(batchSize):
            seqX[ba, :maskX[ba]] = torch.from_numpy(tpl_group[ba]['PSSM'])

        for num in range(self.ModelSize):
            featsize = 10 + self.SS3FeatureModes[num] + \
                self.SS8FeatureModes[num] + self.ACCFeatureModes[num]
            feature = torch.zeros(batchSize, xLen, yLen, featsize)
            for ba in range(batchSize):
                feature[ba, :maskX[ba], :, :] = feature4sequence(
                        tpl_group[ba], self.tgt, self.SS3FeatureModes[num],
                        self.SS8FeatureModes[num], self.ACCFeatureModes[num])
            featdata.append(feature)

        return featdata, seqX, seqY, maskX, maskY
