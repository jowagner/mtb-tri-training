#!/bin/bash

source ~/tri-training/mtb-tri-training/config/locations.sh 

# first test that all patterns yield exactly 1 file

ls -ltr eg-x-z48/prediction-00-E-*-??_*-test-*.conllu.bz2 egox--38/prediction-08-E-*-en_*-test-*.conllu.bz2
echo
ls -ltr hg-x-z38/prediction-00-E-*-??_*-test-*.conllu.bz2 hgox--38/prediction-06-E-*-??_*-test-*.conllu.bz2
echo
ls -ltr ug-x-z48/prediction-00-E-*-??_*-test-*.conllu.bz2 ugox--36/prediction-07-E-*-??_*-test-*.conllu.bz2
echo
ls -ltr vgox-z38/prediction-00-E-*-??_*-test-*.conllu.bz2 vg-x-z48/prediction-03-E-*-??_*-test-*.conllu.bz2

#exit

echo "== en =="

T=egox--38/prediction-08-E-6-en_ewt-test-5iXD9vt4TiqfnsPEZ0qw.conllu.bz2
B=eg-x-z48/prediction-00-E-0-en_ewt-test-nqbGlcGishqEXTIAjeLr.conllu.bz2

~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-test.conllu $B > fasttext-en-baseline-test.correctness
~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-test.conllu $T > fasttext-en-tt-test.correctness
paste fasttext-en-baseline-test.correctness fasttext-en-tt-test.correctness | \
~/tri-training/mtb-tri-training/scripts/mcnemar-exact-test.py

echo "== hu =="

T=hgox--38/prediction-06-E-5-hu_szeged-test-HG7gaLuacTYdj3kB3QXb.conllu.bz2
B=hg-x-z38/prediction-00-E-16-hu_szeged-test-iq356O89sEn3xvQRGu6L.conllu.bz2

~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Hungarian-Szeged/hu_szeged-ud-test.conllu $B > fasttext-hu-baseline-test.correctness
~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Hungarian-Szeged/hu_szeged-ud-test.conllu $T > fasttext-hu-tt-test.correctness
paste fasttext-hu-baseline-test.correctness fasttext-hu-tt-test.correctness | \
~/tri-training/mtb-tri-training/scripts/mcnemar-exact-test.py

echo "== ug =="

T=ugox--36/prediction-07-E-12-ug_udt-test-S24BZ3wApIfLJ7pccipY.conllu.bz2
B=ug-x-z48/prediction-00-E-10-ug_udt-test-S24BZ3wApIfLJ7pccipY.conllu.bz2

~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Uyghur-UDT/ug_udt-ud-test.conllu $B > fasttext-ug-baseline-test.correctness
~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Uyghur-UDT/ug_udt-ud-test.conllu $T > fasttext-ug-tt-test.correctness
paste fasttext-ug-baseline-test.correctness fasttext-ug-tt-test.correctness | \
~/tri-training/mtb-tri-training/scripts/mcnemar-exact-test.py

echo "== vi =="

B=vgox-z38/prediction-00-E-1-vi_vtb-test-S24BZ3wApIfLJ7pccipY.conllu.bz2
T=vg-x-z48/prediction-03-E-10-vi_vtb-test-S24BZ3wApIfLJ7pccipY.conllu.bz2

~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Vietnamese-VTB/vi_vtb-ud-test.conllu $B > fasttext-vi-baseline-test.correctness
~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Vietnamese-VTB/vi_vtb-ud-test.conllu $T > fasttext-vi-tt-test.correctness
paste fasttext-vi-baseline-test.correctness fasttext-vi-tt-test.correctness | \
~/tri-training/mtb-tri-training/scripts/mcnemar-exact-test.py

