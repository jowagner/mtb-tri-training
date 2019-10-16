# Multi-treebank Tri-training

Tri-training dependency parsers with multi-treebank models

Features:
* Sentences and tokens can be selected independently.
* Missing heads and labels for tokens are replaced by random heads and labels.

# Installation

## New to PIP and virtualenv?

```
pip3 install --user --upgrade pip
pip3 install --user virtualenv
```

## Main Tri-training Scripts

```
git clone ssh://git@gogs.adaptcentre.ie:2100/jwagner/mtb-tri-training.git
```

Check under what name the cluster is detected: (You may then need to add a new
section to the file below.)
```
source mtb-tri-training/config/locations.sh
echo $SETTING
```

## Linear Combiner

```
git clone git@github.com:tira-io/ADAPT-DCU.git
```

TODO: make a separate release with first accepted paper

## UDPipe-future

```
git clone git@github.com:CoNLL-UD-2018/UDPipe-Future.git
cd UDPipe-Future/
virtualenv -p /usr/bin/python3.7 venv-udpf
vi venv-udpf/bin/activate
```

Add `LD_LIBRARY_PATH` for a recent CUDA that works with TensorFlow 1.15 to `bin/activate`,
e.g.
`/home/support/nvidia/cuda10/lib64:/home/support/nvidia/cudnn/cuda10_cudnn7_7.5/lib64`.
We used CUDA 10.0 and matching CUDNN.

```
source venv-udpf/bin/activate
pip install tensorflow-gpu==1.14
pip install cython
pip install git+https://github.com/andersjo/dependency_decoding
```

TODO: Could we use the provided script?

## FastText Embeddings for UDPipe-future

TODO:
* pointer to where to get FastText
* current commands to install it.
* steps to train the embeddings

We assume all FastText embeddings in the `.npz` format for UDPipe-future are
in a single folder with filenames `fasttext-xx.npz` where `xx` is a language code.


```
cd UDPipe-Future/
ln -s /spinning/$USER/UDPipe-Future/ud-lowercase-notrain-fasttext.npz fasttext-ga.npz
```

## ELMo For Many Languages

```
git clone git@github.com:HIT-SCIR/ELMoForManyLangs.git

cd ELMoForManyLangs/
virtualenv -p /usr/bin/python3.7 venv-efml
vi venv-efml/bin/activate
```

Add `LD_LIBRARY_PATH` for CUDA 10.1 and matching CUDNN
to `bin/activate`, e.g. `/usr/local/cuda-10.1/lib64`.

```
source venv-efml/bin/activate
pip install torch torchvision
pip install allennlp
cd
mkdir elmo
cd elmo/
ln -s $HOME/tri-training/ELMoForManyLangs/configs/
ln -s /spinning/$USER/elmo/ga_model/
```

TODO: virtualenv example

# Data

The UD Treebank folder structure is assumed, i.e.
* each treebank must be in a separate folder
* the name of a treebank folder must
** start with `UD_` and
** contain, in a lowercased copy of the name, the treebank code (the part after the `_` in the treebank ID)
* each treebank folder must be located directly in the UD treebank folder
* a treebank folder must contain at least a file `xx_yyy-ud-test.conllu` where `xx_yyy` is the treebank ID
* the respective training and development files are `xx_yyy-ud-train.conllu` and `xx_yyy-ud-dev.conllu`

```
cd data/
ln -s /spinning/$USER/ud-treebanks-v2.3/
```

The CoNLL 2017 wikipedia and common crawl data must be uncompressed
(`unxz` the individual `.xz` files after extraction from the provided `.tar` files)
and placed in a folder per language, for example:

```
mkdir CoNLL-2017
cd CoNLL-2017
tar -xf ~/Downloads/Irish-annotated-conll17.tar
mv -i [LR]* Irish/
cd Irish/
unxz *.xz
```

# Configration

`config/locations.sh`

# Training
