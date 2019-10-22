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
* current commands to install it. https://fasttext.cc/docs/en/support.html
* steps to train the embeddings

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

### Train fastext

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

```
for LCODE in ga ug ; do
    echo "== $LCODE =="
    python3 ~/tri-training/UDPipe-Future/embeddings/sources/convert.py \
        --max_words 1000000 model_$LCODE.vec fasttext-$LCODE.npz
done
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
pip install h5py
```

TODO: Do we need to run `python setup.py install`?
I don't think I did but James's readme says so.
The command `python -m elmoformanylangs test`
in `get-elmo-vectors.sh` may simple
work because we `cd`'ed into the efml folder.


```
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

`gen-run-tri-tain-jobs.py` or `tri-train.py --help`
