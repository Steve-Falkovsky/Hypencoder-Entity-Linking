#!/bin/bash
set -ex

mkdir -p corpora_sources
cd corpora_sources

# MedMentions (st21pv)
mkdir -p medmentions/st21pv
wget -O medmentions/st21pv/corpus_pubtator.txt.gz https://github.com/chanzuckerberg/MedMentions/raw/refs/heads/master/st21pv/data/corpus_pubtator.txt.gz
wget -O medmentions/st21pv/corpus_pubtator_pmids_dev.txt https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_dev.txt
wget -O medmentions/st21pv/corpus_pubtator_pmids_test.txt https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_test.txt
wget -O medmentions/st21pv/corpus_pubtator_pmids_trng.txt https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_trng.txt