#!/bin/bash

source ~/tri-training/mtb-tri-training/config/locations.sh 

echo "== en =="

~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-test.conllu ei-x-z3A/prediction-00-E-17-en_ewt-test-3e4xNMNNqnlpkeFpyAt8.conllu.bz2 > mbert-en-baseline-test.correctness
~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-test.conllu eiox-o36/prediction-03-E-14-en_ewt-test-c3pSE13zib4HiUanap4t.conllu.bz2 > mbert-en-tt-test.correctness
paste mbert-en-baseline-test.correctness mbert-en-tt-test.correctness | \
~/tri-training/mtb-tri-training/scripts/mcnemar-exact-test.py

echo "== hu =="

~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Hungarian-Szeged/hu_szeged-ud-test.conllu hiox-z38/prediction-00-E-0-hu_szeged-test-52MOacjtqLQFCfmp8KOc.conllu.bz2 > mbert-hu-baseline-test.correctness
~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Hungarian-Szeged/hu_szeged-ud-test.conllu hiox-z3A/prediction-02-E-2-hu_szeged-test-UchU7krTjwG1nBR5SrzT.conllu.bz2 > mbert-hu-tt-test.correctness
paste mbert-hu-baseline-test.correctness mbert-hu-tt-test.correctness | \
~/tri-training/mtb-tri-training/scripts/mcnemar-exact-test.py

echo "== ug =="

~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Uyghur-UDT/ug_udt-ud-test.conllu uiox-z4A/prediction-00-E-13-ug_udt-test-uGVSr2v6VrMx5OpdZd0I.conllu.bz2 > mbert-ug-baseline-test.correctness
~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Uyghur-UDT/ug_udt-ud-test.conllu uiox-z38/prediction-04-E-11-ug_udt-test-7U5n7gfO8S4LcWJFSNA1.conllu.bz2 > mbert-ug-tt-test.correctness
paste mbert-ug-baseline-test.correctness mbert-ug-tt-test.correctness | \
~/tri-training/mtb-tri-training/scripts/mcnemar-exact-test.py


echo "== vi =="

~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Vietnamese-VTB/vi_vtb-ud-test.conllu viox-o36/prediction-00-E-3-vi_vtb-test-HviBrBdvHAwno4cBXP4U.conllu.bz2 > mbert-vi-baseline-test.correctness
~/tri-training/mtb-tri-training/scripts/conll18_ud_eval.py --tokens ~/data/ud-treebanks-v2.3/UD_Vietnamese-VTB/vi_vtb-ud-test.conllu viox-z3A/prediction-04-E-16-vi_vtb-test-CfC1oy9JOOnosgXwDTrK.conllu.bz2 > mbert-vi-tt-test.correctness
paste mbert-vi-baseline-test.correctness mbert-vi-tt-test.correctness | \
~/tri-training/mtb-tri-training/scripts/mcnemar-exact-test.py

