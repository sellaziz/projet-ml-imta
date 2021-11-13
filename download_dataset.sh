# /bin/bash

cd binary-classification
# Create Folder for the data
mkdir -p data
cd data
echo $PWD
# Banknote Authentication Dataset: https://archive.ics.uci.edu/ml/datasets/banknote+authentication
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt

cd ../../linear-regression
mkdir -p data
cd data
wget https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.info.txt
wget https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data
cd ../..
