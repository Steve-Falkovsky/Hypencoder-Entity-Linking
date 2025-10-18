#!/bin/bash
set -ex

mkdir -p corpora_sources
cd corpora_sources


# BC5CDR
rm -fr CDR_Data
curl -o CDR_Data.zip https://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/CDR_Data.zip
unzip -o CDR_Data.zip
rm CDR_Data.zip

# MeSH 2015 (for BC5CDR)
mkdir -p mesh2015
curl -o mesh2015/desc2015.xml https://nlmpubs.nlm.nih.gov/projects/mesh/2015/xmlmesh/desc2015.xml
curl -o mesh2015/qual2015.xml https://nlmpubs.nlm.nih.gov/projects/mesh/2015/xmlmesh/qual2015.xml
curl -o mesh2015/supp2015.xml https://nlmpubs.nlm.nih.gov/projects/mesh/2015/xmlmesh/supp2015.xml
