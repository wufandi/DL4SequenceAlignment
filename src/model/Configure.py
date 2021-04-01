END = '\033[0m'
BLUE = '\033[94m'


class ModelConfigure():
    def __init__(self, conf):
        self.feat1d = conf['feat1d']
        self.feat2d = 10 + conf['SS3FeatureMode'] + conf['SS8FeatureMode'] + \
            conf['ACCFeatureMode']
        self.layers1d = conf['layers1d']
        self.neurons1d = conf['neurons1d']
        self.layers2d = conf['layers2d']
        self.neurons2d = conf['neurons2d']
        self.dilation = conf['dilation']
        self.seqnet = conf['seqnet']
        self.embedding = conf['embedding']
        self.pairwisenet = conf['pairwisenet']
        self.block = conf['block']
        self.activation = conf['activation']
        self.affine = conf['affine']
        self.track_running_stats = conf['track_running_stats']
        self.SS3FeatureMode = conf['SS3FeatureMode']
        self.SS8FeatureMode = conf['SS8FeatureMode']
        self.ACCFeatureMode = conf['ACCFeatureMode']
        if isinstance(conf['training_set'], dict):
            self.training_set = conf['training_set']
        else:
            self.training_set[conf['training_set']] = 0

    def print_model_parameters(self):
        print(BLUE + "the DRNF model's parameter:" + END)
        print("  Seqnet: %s" % self.seqnet)
        print("      feat1d: %d" % self.feat1d)
        if self.seqnet == "ResNet":
            print("      layers1d:  ", self.layers1d)
            print("      neurons1d: ", self.neurons1d)
        print("  Embedding: %s" % self.embedding)
        print("  Pairwisenet: %s" % self.pairwisenet)
        if self.pairwisenet == "ResNet":
            print("      block: %s" % self.block)
            print("      layers2d:  ", self.layers2d)
            print("      neurons2d: ", self.neurons2d)
            print("      dilation:  ", self.dilation)
        print("  Activation: ", self.activation)
        print("  affine: ", self.affine)
        print("  track_running_stats: ", self.track_running_stats)
        print("  SS3featureMode: %d" % self.SS3FeatureMode)
        print("  SS8featureMode: %d" % self.SS8FeatureMode)
        print("  ACCFeatureMode: %d" % self.ACCFeatureMode)
        return


class ADMMConfigure():
    def __init__(self, conf):
        self.layers2d = conf['layers2d']
        self.neurons2d = conf['neurons2d']
        self.dilation = conf['dilation']
        self.pairwisenet = conf['pairwisenet']
        self.block = conf['block']
        self.activation = conf['activation']
        self.affine = conf['affine']
        self.track_running_stats = conf['track_running_stats']
        if isinstance(conf['training_set'], dict):
            self.training_set = conf['training_set']
        else:
            self.training_set[conf['training_set']] = 0
        self.DRNF = conf['DRNF']
        self.DistFeatureMode = conf['DistFeatureMode']
        self.NormFeatureMode = conf['NormFeatureMode']
        self.AlignFeatureMode = conf['AlignFeatureMode']
        self.FeatSize = conf['DistFeatureMode'] + conf['NormFeatureMode'] + \
            conf['AlignFeatureMode']

    def print_model_parameters(self):
        print(BLUE + "the NDT model's parameter:" + END)
        print("  DRNF model name: %s" % self.DRNF)
        print("  Pairwisenet: %s" % self.pairwisenet)
        if self.pairwisenet == "ResNet":
            print("      block: %s" % self.block)
            print("      layers2d:  ", self.layers2d)
            print("      neurons2d: ", self.neurons2d)
            print("      dilation:  ", self.dilation)
        print("  Activation: ", self.activation)
        print("  affine: ", self.affine)
        print("  track_running_stats: ", self.track_running_stats)
        print("  DistFeatureMode: %d" % self.DistFeatureMode)
        print("  NormFeatureMode: %d" % self.NormFeatureMode)
        print("  AlignFeatureMode: %d" % self.AlignFeatureMode)
        return
