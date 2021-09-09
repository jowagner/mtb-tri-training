#!/bin/bash

source ~/tri-training/mtb-tri-training/config/locations.sh 

# first test that all patterns yield exactly 1 file

ls -ltr eh-x--36/prediction-00-E-*-??_*-test-*.conllu.bz2 ehox--38/prediction-08-E-*-en_*-test-*.conllu.bz2
echo
ls -ltr hh-x-z48/prediction-00-E-*-??_*-test-*.conllu.bz2 hhox-z4A/prediction-03-E-*-??_*-test-*.conllu.bz2
echo
ls -ltr uhox-z48/prediction-00-E-*-??_*-test-*.conllu.bz2 uhox--38/prediction-07-E-*-??_*-test-*.conllu.bz2
echo
ls -ltr vh-x-z38/prediction-00-E-*-??_*-test-*.conllu.bz2 vh-x-z3A/prediction-04-E-*-??_*-test-*.conllu.bz2

#exit

echo "== en =="

B=eh-x--36/prediction-00-E-15-en_ewt-test-S24BZ3wApIfLJ7pccipY.conllu.bz2
T=ehox--38/prediction-08-E-19-en_ewt-test-S24BZ3wApIfLJ7pccipY.conllu.bz2

~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-test.conllu $B > elmo-en-baseline-test.correctness
~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-test.conllu $T > elmo-en-tt-test.correctness
paste elmo-en-baseline-test.correctness elmo-en-tt-test.correctness | \
~/tri-training/mtb-tri-training/scripts/mcnemar-exact-test.py

echo "== hu =="

T=hhox-z4A/prediction-03-E-8-hu_szeged-test-S24BZ3wApIfLJ7pccipY.conllu.bz2
B=hh-x-z48/prediction-00-E-8-hu_szeged-test-T5Go4AImqcNxJH4OjxR8.conllu.bz2

~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Hungarian-Szeged/hu_szeged-ud-test.conllu $B > elmo-hu-baseline-test.correctness
~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Hungarian-Szeged/hu_szeged-ud-test.conllu $T > elmo-hu-tt-test.correctness
paste elmo-hu-baseline-test.correctness elmo-hu-tt-test.correctness | \
~/tri-training/mtb-tri-training/scripts/mcnemar-exact-test.py

echo "== ug =="

T=uhox--38/prediction-07-E-4-ug_udt-test-S24BZ3wApIfLJ7pccipY.conllu.bz2
B=uhox-z48/prediction-00-E-18-ug_udt-test-S24BZ3wApIfLJ7pccipY.conllu.bz2

~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Uyghur-UDT/ug_udt-ud-test.conllu $B > elmo-ug-baseline-test.correctness
~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Uyghur-UDT/ug_udt-ud-test.conllu $T > elmo-ug-tt-test.correctness
paste elmo-ug-baseline-test.correctness elmo-ug-tt-test.correctness | \
~/tri-training/mtb-tri-training/scripts/mcnemar-exact-test.py

echo "== vi =="

T=vh-x-z3A/prediction-04-E-13-vi_vtb-test-S24BZ3wApIfLJ7pccipY.conllu.bz2
B=vh-x-z38/prediction-00-E-1-vi_vtb-test-ZDsy73gGOyab9OEEoZef.conllu.bz2

~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Vietnamese-VTB/vi_vtb-ud-test.conllu $B > elmo-vi-baseline-test.correctness
~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Vietnamese-VTB/vi_vtb-ud-test.conllu $T > elmo-vi-tt-test.correctness
paste elmo-vi-baseline-test.correctness elmo-vi-tt-test.correctness | \
~/tri-training/mtb-tri-training/scripts/mcnemar-exact-test.py

