# DL4SequenceAlignment #

### Introduction

This package has the source code for DRNF and NDThreader for protein threading.
1. DRNF: protein threading by deep learning when inter-residue distance information is not used
2. NDThreader: protein threading by deep learning with distance potential

This package is only for non-commerical use. 

## set environmental variable

    export OMP_NUM_THREADS=1

The OMP_NUM_THREADS environment variable sets the number of threads to be used for parallel execution of some modules.

If you want to use multiple CPUs, you must set it to be larger than 1 especially for following scripts:
DRNFSearch.py, NDThreaderSearch.py and BatchRNDTAlign.py

### Download package
This code is also available at http://raptorx.uchicago.edu/Download/ .
The pre-computed template files are available at http://raptorx.uchicago.edu/download/ .

### Set up
Some tools are written in C/C++. To use them, you may build them by running the shell scripts setup.sh by

    ./setup.sh

** Required packages:

1) python >= 3.6

2) numpy >= 1.18.5

3) pytorch >= 1.1.0

4) biopython >= 1.75

5) sharedmem >= 0.3.7

6) tqdm >= 4.46.1

### Run

1. To predict a pairwise sequence-template alignment by DRNF, you may use DRNFAlign.py. For example, you may run

    python DRNFAlign.py \
        -m params/model.DA.SSA.1.pth \
        -t example/2gnqA.tpl.pkl \
        -q example/T0954.tgt

Where model.DA.SSA.1.pth is one pre-trained deep model. All pre-trained DRNF models are available in params/.
The input is a template file (ending with .tpl or .tpl.pkl) and a query sequence feature file (ending with .tgt or .tgt.pkl)

2. To thread a query protein sequence to a list of templates by DRNF, you may use DRNFSearch.py. For example, you may run

    python DRNFSearch.py \
        -l example/T0954.template.txt \
        -m params/model.DA.SSA.1.pth params/model.DA.SSA.2.pth \
        -q example/T0954.tgt \
        -t database/TPL_BC100 -o T0954

Where T0954.template.txt contains a list of templates to be used and database/TPL_BC100 shall contain all the template files in T0954.template.txt.
"-o" specifies the output folder.

3. To predict a sequence-template alignment by NDThreader with the ADMM algorithm, you may use NDThreaderAlign.py. For example, you can run

    python NDThreaderAlign.py \
        -q example/T0954.tgt -t example/2gnqA.tpl.pkl \
        -d example/T0954.distPotential.DFIRE16.pkl \
        -m params/model.DA.SSA.1.pth params/model.DA.SSA.2.pth  \
        -i 10 -w 1

Where T0954.distPotential.DFIRE16.pkl is the distance potential predicted by RaptorX-3DModeling for the query protein T0954.
"-w" specifies the weight of singleton features including sequence profile and secondary structure.
"-i" specifies the number of iterations used by ADMM.

4. To thread a sequence by NDThreader with the ADMM algorithm, you can use NDThreaderSearch.py. For example, you can run

    python NDThreaderSearch.py \
        -l example/T0954.template.txt \
        -m params/model.DA.SSA.1.pth params/model.DA.SSA.2.pth \
           params/model.TM.SSA.1.pth \
        -q example/T0954.tgt -d example/T0954.distPotential.DFIRE16.pkl \
        -t database/TPL_BC100 -o T0954

5. To predict alignments by NDThreader with deep ResNet, you should use BatchRNDTAlign.py. For example, you can run

    python BatchRNDTAlign.py -m params/RNDTmodel.DA.1.pth \
        -l example/alignment_list \
        -t database/TPL_BC100 -q database/TGT_BC100 \
        -d database/DIST_BC100  -o output

Where alignment_list contains a list of template-sequence pairs. Note that BatchRNDTAlign.py can only be used with pre-trained model files starting with "RNDTmodel".
database/TGT_BC100 is the folder for all query sequence feature files and database/DIST_BC100 is the folder for predicted distance potential. 

### RaptorX-3DModeling

If you want to generate feature files for query sequences and templates by yourself, you may use some scripts in the RaptorX-3DModeling package at https://github.com/j3xugit/RaptorX-3DModeling/.
To generate a sequence feature file from an MSA, you may use RaptorX-3DModeling/Common/MSA2TGT.sh or RaptorX-3DModeling/BuildFeatures/GenTGTFromA3M.sh;
Note that MSA2TGT.sh needs Theano, but GenTGTFromA3M.sh do not.

To predict inter-residue distance probability distribution, you may use RaptorX-3DModeling/DL4DistancePrediction4/Scripts/PredictPairRelationFromMSA.sh, which needs Theano but not PyRosetta.
To convert distance probability distribution to distance potential, you may use RaptorX-3DModeling/DL4DistancePrediction4/Scripts/DeriveDistInfo4Threading.sh.

Currently this version of RaptorX-3DModeling uses Python2 and Theano. A new version using Python3 and Tensorflow will be released soon.

### More tools ###
all tools are in tools/
1. Alignment_Comparison: compare two alignments, which can be used to compute recall and precison when one alignment is the ground truth.

## Reference
Fandi Wu and Jinbo Xu. Deep Template-based Protein Structure Prediction. PLoS Computational Biology, 2021.
Also appears at https://www.biorxiv.org/content/10.1101/2020.12.26.424433v1.full 

### Contact
Fandi Wu wufandi@outlook.com

Jinbo Xu jinboxu@gmail.com
