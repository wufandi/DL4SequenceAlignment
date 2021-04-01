#!/bin/bash

# C++ part

if [ ! -d tools ]
then
    mkdir tools
fi
g++ utils/Alignment_Comparison.cpp -o tools/Alignment_Comparison
g++ utils/distance.cpp -o src/model/distance.so -shared -fPIC
g++ utils/crf.cpp -o src/model/crf.so -shared -fPIC

echo "finish set up ! "
