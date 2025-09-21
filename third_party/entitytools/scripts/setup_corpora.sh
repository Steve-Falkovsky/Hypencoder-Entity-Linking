#!/bin/bash
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p corpora/medmentions
cd corpora/medmentions

# MedMentions
rm -f corpus_pubtator.txt.gz corpus_pubtator_pmids_dev.txt corpus_pubtator_pmids_test.txt corpus_pubtator_pmids_trng.txt
wget -O corpus_pubtator.txt.gz https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator.txt.gz
wget -O corpus_pubtator_pmids_dev.txt https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_dev.txt
wget -O corpus_pubtator_pmids_test.txt https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_test.txt
wget -O corpus_pubtator_pmids_trng.txt https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_trng.txt

python $SCRIPT_DIR/prepare_medmentions.py --medmentions_dir . --out_train medmentions_train.bioc.xml.gz --out_val medmentions_val.bioc.xml.gz --out_test medmentions_test.bioc.xml.gz
