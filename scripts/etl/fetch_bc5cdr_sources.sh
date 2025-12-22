#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_RAW="$ROOT/data/raw"
CORPORA_DIR="$DATA_RAW/corpora_sources"

mkdir -p "$CORPORA_DIR"
cd "$CORPORA_DIR"


# BC5CDR
rm -fr CDR_Data
curl -L -o CDR_Data.zip https://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/CDR_Data.zip
unzip -o CDR_Data.zip
rm CDR_Data.zip

# MeSH 2015 (for BC5CDR)
mkdir -p mesh2015
curl -L -o mesh2015/desc2015.xml https://nlmpubs.nlm.nih.gov/projects/mesh/2015/xmlmesh/desc2015.xml
curl -L -o mesh2015/qual2015.xml https://nlmpubs.nlm.nih.gov/projects/mesh/2015/xmlmesh/qual2015.xml
curl -L -o mesh2015/supp2015.xml https://nlmpubs.nlm.nih.gov/projects/mesh/2015/xmlmesh/supp2015.xml
