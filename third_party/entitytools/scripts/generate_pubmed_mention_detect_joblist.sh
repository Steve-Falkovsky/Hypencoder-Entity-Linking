#!/bin/bash
set -ex

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 MODEL OUTDIR OUTSCRIPT"
    exit 1
fi

MODEL=$1
OUTDIR=$2
OUTSCRIPT=$3

mkdir -p $OUTDIR
MODEL=$(readlink -f $MODEL)
OUTDIR=$(readlink -f $OUTDIR)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


FTPPATH=ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline

rm -f $OUTSCRIPT

for FILENAME in $(curl --silent $FTPPATH/ | grep -oP "pubmed\w+.xml.gz" | sort -u)
do
    BASENAME=$(echo $FILENAME | grep -oP "pubmed\w+")


    echo "bash $SCRIPT_DIR/download_pubmed_and_mention_detect.sh $FTPPATH/$FILENAME $MODEL $OUTDIR" >> $OUTSCRIPT
done
