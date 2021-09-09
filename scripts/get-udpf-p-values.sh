#!/bin/bash

source ~/tri-training/mtb-tri-training/config/locations.sh 

# first test that all patterns yield one exactly 1 file

ls -ltr ef-x-z4A/prediction-00-E-*-??_*-test-*.conllu.bz2 efox--38/prediction-08-E-*-en_*-test-*.conllu.bz2
echo
ls -ltr hfox--36/prediction-00-E-*-??_*-test-*.conllu.bz2 hfox--36/prediction-12-E-*-??_*-test-*.conllu.bz2
echo
ls -ltr uf-x-z48/prediction-00-E-*-??_*-test-*.conllu.bz2 uf-x-z38/prediction-08-E-*-??_*-test-*.conllu.bz2
echo
ls -ltr vfox-z4A/prediction-00-E-*-??_*-test-*.conllu.bz2 vf-x--38/prediction-08-E-*-??_*-test-*.conllu.bz2

#exit


echo "== en =="

T=efox--38/prediction-08-E-11-en_ewt-test-gF2C0owMHRku08Z7VgrK.conllu.bz2
B=ef-x-z4A/prediction-00-E-1-en_ewt-test-sdvnSNNDlBp5BnaydvQN.conllu.bz2

~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-test.conllu $B > udpf-en-baseline-test.correctness
~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-test.conllu $T > udpf-en-tt-test.correctness
paste udpf-en-baseline-test.correctness udpf-en-tt-test.correctness | \
~/tri-training/mtb-tri-training/scripts/mcnemar-exact-test.py

echo "== hu =="

B=hfox--36/prediction-00-E-6-hu_szeged-test-fIeJ4ywd4gIHLsAOs4Fv.conllu.bz2
T=hfox--36/prediction-12-E-7-hu_szeged-test-T8E2EPE5HXCc65Pajb5n.conllu.bz2

~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Hungarian-Szeged/hu_szeged-ud-test.conllu $B > udpf-hu-baseline-test.correctness
~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Hungarian-Szeged/hu_szeged-ud-test.conllu $T > udpf-hu-tt-test.correctness
paste udpf-hu-baseline-test.correctness udpf-hu-tt-test.correctness | \
~/tri-training/mtb-tri-training/scripts/mcnemar-exact-test.py

echo "== ug =="

B=uf-x-z48/prediction-00-E-7-ug_udt-test-XvabQt9xv2BCRuYveVU4.conllu.bz2
T=uf-x-z38/prediction-08-E-3-ug_udt-test-BfM9C00F1zg2BOGUjqYB.conllu.bz2

~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Uyghur-UDT/ug_udt-ud-test.conllu $B > udpf-ug-baseline-test.correctness
~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Uyghur-UDT/ug_udt-ud-test.conllu $T > udpf-ug-tt-test.correctness
paste udpf-ug-baseline-test.correctness udpf-ug-tt-test.correctness | \
~/tri-training/mtb-tri-training/scripts/mcnemar-exact-test.py


echo "== vi =="

T=vf-x--38/prediction-08-E-6-vi_vtb-test-S24BZ3wApIfLJ7pccipY.conllu.bz2
B=vfox-z4A/prediction-00-E-20-vi_vtb-test-S24BZ3wApIfLJ7pccipY.conllu.bz2

~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Vietnamese-VTB/vi_vtb-ud-test.conllu $B > udpf-vi-baseline-test.correctness
~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Vietnamese-VTB/vi_vtb-ud-test.conllu $T > udpf-vi-tt-test.correctness
paste udpf-vi-baseline-test.correctness udpf-vi-tt-test.correctness | \
~/tri-training/mtb-tri-training/scripts/mcnemar-exact-test.py

