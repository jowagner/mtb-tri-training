# Tri-training for Dependency Parsers

This repository contains an implementation of tri-training and glue code
run the experiments of the paper

Joachim Wagner and Jennifer Foster (to appear):
Revisiting Tri-training of Dependency Parsers.
In Proceedings of The 2021 Conference on Empirical Methods
in Natural Language Processing (EMNLP 2021),
7th to 11th November 2021,
Online and in the Barceló Bávaro Convention Centre, Punta Cana, Dominican Republic.
Association for Computational Linguistics

Features:
* Supports different ways to sample the seed data
* Combine automatically labelled data from different iterations
* Parse a new sample of the unlabelled data in each iteration
* Wrapper modules for [UDPipe-Future](https://github.com/CoNLL-UD-2018/UDPipe-Future) with external FastText, ELMo and Multilingual BERT word embeddings
* Use [linear tree combiner](https://github.com/jowagner/ud-combination) as the ensemble and average evaluation scores over multiple runs of the non-deterministic combiner

Untested features:
* Modular to support other tasks than UD dependency parsing
* Parse as much unlabelled data as needed to reach a target size of teaching material
* Training with more than 3 learners
* Allow (small) disagreements between teachers
* Require disagreement of the learner
* Sentences and tokens can be selected independently
* Missing heads and labels for tokens are replaced by random heads and labels

# Installation

## New to PIP and virtualenv?

This can be skipped on ICHEC. Due to missing Python header files, some
non-binary pip packages fail to compile.

```
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
pip3 install --user virtualenv
```

Append to `.bashrc` and re-login:
```
# for our own pip and virtualenv:
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib
```

## Main Tri-training Scripts

```
mkdir tri-training
cd tri-training
git clone ssh://git@gogs.adaptcentre.ie:2100/jwagner/mtb-tri-training.git
```

TODO: make repo public when paper is accepted and made available

Check under what name the cluster is detected: (You may then need to add a new
section to the file below.)
```
source mtb-tri-training/config/locations.sh
echo $SETTING
```

To not have to set `PRJ_DIR` before running any of the tri-training scripts,
let `$HOME/mtb-tri-training` point to the repository:
```
cd
ln -s /path/to/tri-training/mtb-tri-training/
```

## Linear Combiner

```
git clone git@github.com:tira-io/ADAPT-DCU.git
```

TODO: update with https://github.com/jowagner/ud-combination

## UDPipe-future

### Using Virtualenv

```
git clone git@github.com:CoNLL-UD-2018/UDPipe-Future.git
cd UDPipe-Future/
virtualenv -p /usr/bin/python3.7 venv-udpf
vi venv-udpf/bin/activate
```

Note: Some of our experiments were run on a second cluster using Python 3.6 instead of 3.7.

Add `LD_LIBRARY_PATH` for a recent CUDA with CuDNN
that works with TensorFlow 1.14 to `bin/activate`,
e.g.
on the ADAPT clusters:
```
LD_LIBRARY_PATH=/home/support/nvidia/cuda10/lib64:/home/support/nvidia/cudnn/cuda10_cudnn7_7.5/lib64:"$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH
```

As in the above configuration, we used CUDA 10.0 and matching CuDNN.
TODO: Why do we not use `UDPIPE_FUTURE_LIB_PATH` in `config/locations.sh`?

```
source venv-udpf/bin/activate
pip install tensorflow-gpu==1.14
pip install cython
pip install git+https://github.com/andersjo/dependency_decoding
```

TODO: Could we use the script provided by udpipe-future? What is the `venv` module that it uses?

### Using Conda

On ICHEC, conda needs to be loaded first:
```
module load conda/2
```

Then:
```
conda create --name udpf python=3.7 \
    tensorflow-gpu==1.14 \
    cython
```

If this is the first time using conda on ICHEC, you need to run
`conda init bash`, re-login, run `conda config --set auto_activate_base false`
and re-login again.

TODO: ICHEC support says to use `source activate udpf` instead, not
requiring initialisation. Test this on next install. (This will also
require adjustments to most of the shell scripts
`mtb-tri-training/scripts/*.sh`.

Then:
```
conda activate udpf
pip install git+https://github.com/andersjo/dependency_decoding
conda deactivate
```

If CUDA and CuDNN libraries are not in your library path already, you need to
set `UDPIPE_FUTURE_LIB_PATH` in `config/locations.sh`, e.g.
on ICHEC:
```
UDPIPE_FUTURE_LIB_PATH="/ichec/packages/cuda/10.0/lib64":"$HOME/cudnn-for-10.0/lib64"
```

## FastText Embeddings for UDPipe-future

Needed only if not re-using the fasttext `.npz` word embeddings
of our experiment.

### Installation

As of writing,
(FastText)[https://fasttext.cc/docs/en/support.html] is installed by
cloning the repository, `cd`-ing into it, running `make` and copying
the `fasttext` binary to a folder in your `PATH`, e.g. `$HOME/.local/bin`.
As the binary is built with `-march=native`, it needs to be built
on a machine supporting all desired CPU features, e.g. AVX.

### Extract tokenised text

Adjust and run the following script. Set `TMP_DIR` inside the script to a
suitable location with at least 240 GB free space if, as often, your `/tmp`
is smaller than that.
```
run-get-conllu-text.sh
```

For Irish, we observed that the udpipe tokeniser fails to separate neutral
double quotes as they do not occur in the treebank. However, for consistency
with other languages, we do not address this issue here.

We use truecase as UDpipe-future recently added supports both truecase and
we expect the character-based fasttext embeddings to learn the relationship
between lowercase and uppercase letters.

### Train FastText

As Straka et al. (2019), we run fasttext with `-minCount 5 -epoch 10 -neg 10`, e.g.
```
fasttext skipgram -minCount 5 -epoch 10 -neg 10 -input Irish.txt -output model_ga
```

Job script example: `run-fasttext-en.job`
(Note that fasttext uses only a few GB of RAM. We request a lot of RAM
only as a workaround to request a CPU with AVX support as year of purchase
and amount of RAM are correlated. The English model takes about 2 1/2 days
to train. The Uyghur model takes only a few minutes.)


### Conversion to UDPipe-future .npz format

The `.vec` files can be converted with `convert.py` provided with
UDPipe-future.
We assume all FastText embeddings in the `.npz` format for UDPipe-future are
in a single folder with filenames `fasttext-xx.npz` where `xx` is a language code.
As Straka et al. (2019), we limit the vocabulary to the 1 million most frequent
types.

```
for LCODE in ga ug ; do
    echo "== $LCODE =="
    python3 ~/tri-training/UDPipe-Future/embeddings/sources/convert.py \
        --max_words 1000000 model_$LCODE.vec fasttext-$LCODE.npz
done
```

## ELMo For Many Languages

https://github.com/HIT-SCIR/ELMoForManyLangs

```
git clone git@github.com:HIT-SCIR/ELMoForManyLangs.git
```

### Using Virtualenv

```
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
pip install h5py
```

### Using Conda on ICHEC

```
module load conda/2
conda create --name efml python=3.7 h5py
conda activate efml
pip install torch torchvision
pip install allennlp
conda deactivate
```

### Module Installtion not Required

It is not necessary to run `python setup.py install`:
The command `python -m elmoformanylangs test`
in `get-elmo-vectors.sh` work because we `cd`
into the efml folder.

### Models

After extracting the elmoformanylangs model files, the
`config_path` variable in the `config.json` files has
to be adjusted.

```
mkdir ug_model
cd ug_model
unzip ../downloads/175.zip
vi config.json
```

We assume that the elmo configuration and models are in a single
folder and to be able to re-use the same `.json` files
on different systems, we use symlinks:

```
cd
mkdir elmo
cd elmo/
ln -s $HOME/tri-training/ELMoForManyLangs/configs/
ln -s /spinning/$USER/elmo/ga_model/
```

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
for L in Irish Uyghur Hungarian Vietnamese English ; do
    echo == $L ==
    tar -xf ${L}*.tar
    mv -i LICENSE* README $L/
done
```

To use this data directly, uncompress the `.xz` files. However, as the ouput
is very big, we remove `#` comments and predicted fields from the unlabelled data
and, for the larger data sets, stochastically reduce the dataset to a fraction.
Helper script: `run-clean-unlabelled.sh`

The tri-training script loads CoNNL-2017 Wikipedia data for language `xx` if the
treebank ID `xx_wp17` is used and CoNNL-2017 Common Crawl data for `xx_cc17`.

# Configration

`config/locations.sh`

# Training

* `gen-run-tri-tain-jobs.py` writes job files to a new folder `jobs`
* `tri-train.py --help`
