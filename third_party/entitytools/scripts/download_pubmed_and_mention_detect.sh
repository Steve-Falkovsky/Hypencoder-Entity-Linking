#!/bin/bash
set -eux

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 URL MODEL OUTDIR"
    exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

URL=$1
MODEL=$2
OUTDIR=$3

mkdir -p /tmp/pubmed
mkdir -p $OUTDIR

BASENAME=$(echo $URL | grep -oP "pubmed\w+")

PUBMED_FILE=/tmp/pubmed/$BASENAME.xml.gz
NER_FILE=$OUTDIR/$BASENAME.bioc.xml.gz

if [ -f $NER_FILE ]; then
    echo "Skipping $NER_FILE"
    continue
fi

echo $URL
wget -O $PUBMED_FILE $URL

python $SCRIPT_DIR/apply_mention_detector.py --input_pubmed $PUBMED_FILE --mention_detector_model $MODEL --output_bioc $NER_FILE --mark_sentences
#python apply_exact_string_matcher.py --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --input_pubmed $pubmed_file --output_bioc $ner_file

touch $NER_FILE.done

rm $PUBMED_FILE
