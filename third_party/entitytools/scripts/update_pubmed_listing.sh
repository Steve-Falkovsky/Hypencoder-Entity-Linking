#!/bin/bash
set -ex

echo "Updating PubMed Listing"

ftpPath=ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/

curl --silent $ftpPath |\
grep -oP "pubmed\w+.xml.gz" |\
sort -u |\
awk -v ftpPath=$ftpPath ' { print ftpPath$0 } ' > pubmed_listing.txt
