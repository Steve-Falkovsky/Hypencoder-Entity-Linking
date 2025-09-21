from bioc import pubtator
import gzip
import argparse
import csv
from tqdm.auto import tqdm
import os
import json

from entitytools.file_formats import pubtator_to_bioc, save_bioc_docs

def main():
    parser = argparse.ArgumentParser(description='Convert NCBI Disease to BioCXML and set up a matching MEDIC ontology')
    parser.add_argument('--ncbidisease_dir',required=True,type=str,help='Directory with source NCBI Disease corpus files')
    parser.add_argument('--ctd_medic_diseases',required=True,type=str,help='CTD Medic Disease file')
    parser.add_argument('--out_train',required=True,type=str,help='Output Gzipped BioC XML for training data')
    parser.add_argument('--out_val',required=True,type=str,help='Output Gzipped BioC XML for validation data')
    parser.add_argument('--out_test',required=True,type=str,help='Output Gzipped BioC XML for test data')
    parser.add_argument('--out_ontology',required=True,type=str,help='Output Gzipped JSON file with MEDIC ontology')
    args = parser.parse_args()

    assert os.path.isdir(args.ncbidisease_dir)

    print("Loading documents...")
    with open(f"{args.ncbidisease_dir}/NCBItrainset_corpus.txt") as fp:
        train_docs = pubtator.load(fp)
    with open(f"{args.ncbidisease_dir}/NCBIdevelopset_corpus.txt") as fp:
        val_docs = pubtator.load(fp)
    with open(f"{args.ncbidisease_dir}/NCBItestset_corpus.txt") as fp:
        test_docs = pubtator.load(fp)

    train_docs = [ pubtator_to_bioc(doc) for doc in train_docs ]
    val_docs = [ pubtator_to_bioc(doc) for doc in val_docs ]
    test_docs = [ pubtator_to_bioc(doc) for doc in test_docs ]

    print("Loading and reformatting MEDIC...")
    ontology = []
    with gzip.open(args.ctd_medic_diseases,'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm(reader):
            if row[0].startswith('#'):
                continue

            # Newer CTD format has eight columns, but older has seven (missing slim_mappings)
            #disease_name, disease_id, alt_disease_ids, definition, parent_ids, tree_numbers, parent_tree_numbers, synonyms, slim_mappings = row
            disease_name, disease_id, alt_disease_ids, definition, parent_ids, tree_numbers, parent_tree_numbers, synonyms = row
    
            entity = {'id':disease_id.replace('MESH:',''),
    				'name':disease_name,
    				'aliases':sorted(set(synonyms.split('|'))),
    				'definition':definition }
    
            ontology.append(entity)

    print("Filtering annotations for those in ontology...")
    ontology_by_id = {e['id']:e for e in ontology}
    for doc in train_docs + val_docs + test_docs:
        for passage in doc.passages:
            passage.annotations = [ anno for anno in passage.annotations if anno.infons['concept_id'] in ontology_by_id ]

    print("Saving ontology...")
    with gzip.open(args.out_ontology,'wt') as f:
        json.dump(ontology,f)
        
    print("Saving documents...")
    save_bioc_docs(train_docs, args.out_train)
    save_bioc_docs(val_docs, args.out_val)
    save_bioc_docs(test_docs, args.out_test)

    print(f"{len(train_docs)=} {len(val_docs)=} {len(test_docs)=}")
    print("Done")
	
if __name__ == '__main__':
	main()


