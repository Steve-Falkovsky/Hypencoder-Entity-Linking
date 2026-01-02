from bioc import biocxml
from pathlib import Path
import gzip
import argparse
import json
import xml.etree.ElementTree as ET



"""
Command to run this (defaults assume data/raw/corpora_sources is populated by fetch_bc5cdr_sources.sh):
python scripts/etl/prepare_bc5cdr_and_mesh.py --bc5cdr_dir data/raw/corpora_sources/CDR_Data/CDR.Corpus.v010516 --mesh2015_dir data/raw/corpora_sources/mesh2015 --out_train data/processed/bc5cdr_train.bioc.xml.gz --out_val data/processed/bc5cdr_val.bioc.xml.gz --out_test data/processed/bc5cdr_test.bioc.xml.gz --out_ontology data/processed/mesh2015.json.gz

"""

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

DEFAULT_BC5CDR_DIR = DATA_RAW / "corpora_sources/CDR_Data/CDR.Corpus.v010516"
DEFAULT_MESH2015_DIR = DATA_RAW / "corpora_sources/mesh2015"
DEFAULT_OUT_TRAIN = DATA_PROCESSED / "bc5cdr_train.bioc.xml.gz"
DEFAULT_OUT_VAL = DATA_PROCESSED / "bc5cdr_val.bioc.xml.gz"
DEFAULT_OUT_TEST = DATA_PROCESSED / "bc5cdr_test.bioc.xml.gz"
DEFAULT_OUT_ONTOLOGY = DATA_PROCESSED / "mesh2015.json.gz"


def main():
    parser = argparse.ArgumentParser(description='Convert BC5CDR corpus to BioCXML and set up a matching MeSH ontology')
    parser.add_argument('--bc5cdr_dir',default=DEFAULT_BC5CDR_DIR,type=Path,help='Directory with source NCBI Disease corpus files')
    parser.add_argument('--mesh2015_dir',default=DEFAULT_MESH2015_DIR,type=Path,help='Directory containing MeSH 2015 XML files')
    parser.add_argument('--out_train',default=DEFAULT_OUT_TRAIN,type=Path,help='Output Gzipped BioC XML for training data')
    parser.add_argument('--out_val',default=DEFAULT_OUT_VAL,type=Path,help='Output Gzipped BioC XML for validation data')
    parser.add_argument('--out_test',default=DEFAULT_OUT_TEST,type=Path,help='Output Gzipped BioC XML for test data')
    parser.add_argument('--out_ontology',default=DEFAULT_OUT_ONTOLOGY,type=Path,help='Output Gzipped JSON file with MeSH ontology')
    args = parser.parse_args()
    
    if not args.bc5cdr_dir.is_dir():
        parser.error(f"BC5CDR directory not found: {args.bc5cdr_dir}")
    if not args.mesh2015_dir.is_dir():
        parser.error(f"MeSH2015 directory not found: {args.mesh2015_dir}")

    # Ensure output directories exist (e.g., data/processed)
    for out_path in (args.out_train, args.out_val, args.out_test, args.out_ontology):
        out_path.parent.mkdir(parents=True, exist_ok=True)


    # ----------------- BC5CDR processing -----------------
    print("Loading documents...")
    with open(args.bc5cdr_dir / 'CDR_TrainingSet.BioC.xml') as f:
        train_collection = biocxml.load(f)
    with open(args.bc5cdr_dir / 'CDR_DevelopmentSet.BioC.xml') as f:
        val_collection = biocxml.load(f)
    with open(args.bc5cdr_dir / 'CDR_TestSet.BioC.xml') as f:
        test_collection = biocxml.load(f)

    print("Reformatting BioC XML annotations...")
    for doc in train_collection.documents + val_collection.documents + test_collection.documents:
        for passage in doc.passages:
            passage.annotations = [ anno for anno in passage.annotations if anno.infons['MESH'] != '-1' and '|' not in anno.infons['MESH'] ]
            for anno in passage.annotations:
                anno.infons = { 'concept_id': f"MESH:{anno.infons['MESH']}" }

    print("Removing non-contiguous annotations...")
    for doc in train_collection.documents + val_collection.documents + test_collection.documents:
        for passage in doc.passages:
            passage.annotations = [ anno for anno in passage.annotations if len(anno.locations) == 1 ]
    
    
    
    # --------------- MeSH2015 processing -----------------
    ontology = []

    print("Loading MeSH descriptors...")
    tree = ET.parse(args.mesh2015_dir / "desc2015.xml")
    root = tree.getroot()
    for record in root.findall("DescriptorRecord"):
        record_ui = record.find("DescriptorUI").text
        name = record.find("DescriptorName/String").text
        
        scope_note = record.find("ConceptList/Concept/ScopeNote")
        scope_note = scope_note.text if scope_note is not None else ''
        
        aliases = [ s.text for s in record.findall("ConceptList/Concept/TermList/Term/String") ]
        aliases = sorted(set(aliases))
    
        entity = {'id':f'MESH:{record_ui}',
                  'name':name,
                  'aliases':aliases,
                  'definition':scope_note }
    
        ontology.append(entity)
    
    print("Loading MeSH supplementary concepts...")
    tree = ET.parse(args.mesh2015_dir / "supp2015.xml")
    root = tree.getroot()
    for record in root.findall("SupplementalRecord"):
        record_ui = record.find("SupplementalRecordUI").text
        name = record.find("SupplementalRecordName/String").text
        
        aliases = [ s.text for s in record.findall("ConceptList/Concept/TermList/Term/String") ]
        aliases = sorted(set(aliases))
    
        entity = {'id':f'MESH:{record_ui}',
                  'name':name,
                  'aliases':aliases,
                  'definition':'' }
    
        ontology.append(entity)

    print("Filtering annotations for those in ontology...")
    ontology_by_id = {e['id']:e for e in ontology}
    for doc in train_collection.documents + val_collection.documents + test_collection.documents:
        for passage in doc.passages:
            for anno in passage.annotations:
                assert anno.infons['concept_id'] in ontology_by_id

    print("Saving ontology...")
    with gzip.open(args.out_ontology,'wt') as f:
        json.dump(ontology,f)
        
    print("Saving documents...")    
    with gzip.open(args.out_train, 'wt', encoding='utf8') as f:
        biocxml.dump(train_collection, f)
    with gzip.open(args.out_val, 'wt', encoding='utf8') as f:
        biocxml.dump(val_collection, f)
    with gzip.open(args.out_test, 'wt', encoding='utf8') as f:
        biocxml.dump(test_collection, f)

    print(f"{len(train_collection.documents)=} {len(val_collection.documents)=} {len(test_collection.documents)=}")
    print("Done")
	
if __name__ == '__main__':
	main()


