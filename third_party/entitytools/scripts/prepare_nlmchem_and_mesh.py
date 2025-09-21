from bioc import biocxml
import gzip
import argparse
import os
import json
import xml.etree.ElementTree as ET

from entitytools.file_formats import save_bioc_docs

def main():
    parser = argparse.ArgumentParser(description='Convert NLM-Chem corpus to BioCXML and set up a matching MeSH ontology')
    parser.add_argument('--nlmchem_dir',required=True,type=str,help='Directory with source NCBI Disease corpus files')
    parser.add_argument('--mesh2021_dir',required=True,type=str,help='Directory containing MeSH 2021 XML files')
    parser.add_argument('--out_train',required=True,type=str,help='Output Gzipped BioC XML for training data')
    parser.add_argument('--out_val',required=True,type=str,help='Output Gzipped BioC XML for validation data')
    parser.add_argument('--out_test',required=True,type=str,help='Output Gzipped BioC XML for test data')
    parser.add_argument('--out_ontology',required=True,type=str,help='Output Gzipped JSON file with MeSH ontology')
    args = parser.parse_args()

    assert os.path.isdir(args.nlmchem_dir)

    print("Loading data split...")
    with open(f'{args.nlmchem_dir}/pmcids_train.txt') as f:
        train_pmcids = [ line.strip() for line in f ]
    with open(f'{args.nlmchem_dir}/pmcids_dev.txt') as f:
        val_pmcids = [ line.strip() for line in f ]
    with open(f'{args.nlmchem_dir}/pmcids_test.txt') as f:
        test_pmcids = [ line.strip() for line in f ]

    print("Loading documents...")
    train_docs, val_docs, test_docs = [], [], []
    for filename in os.listdir(f"{args.nlmchem_dir}/ALL"):
        with open(f"{args.nlmchem_dir}/ALL/{filename}") as fp:
            collection = biocxml.load(fp)
            for doc in collection.documents:
                if doc.id in train_pmcids:
                    train_docs.append(doc)
                elif doc.id in val_pmcids:
                    val_docs.append(doc)
                elif doc.id in test_pmcids:
                    test_docs.append(doc)
                else:
                    raise RuntimeError(f"{doc.id=} is not in one of the train/val/test groupings")

    print("Reformatting BioC XML annotations...")
    for doc in train_docs+val_docs+test_docs:
        for passage in doc.passages:
            passage.annotations = [ anno for anno in passage.annotations if anno.infons['type'] == 'Chemical']
            for anno in passage.annotations:
                anno.infons = { 'concept_id': anno.infons['identifier'] }

    ontology = []

    print("Loading MeSH descriptors...")
    tree = ET.parse(f"{args.mesh2021_dir}/desc2021.xml")
    root = tree.getroot()
    for record in root.findall("DescriptorRecord"):
        record_ui = record.find("DescriptorUI").text
        name = record.find("DescriptorName/String").text
    
        scope_note = record.find("ConceptList/Concept/ScopeNote")
        scope_note = scope_note.text if scope_note else ''
        
        aliases = [ s.text for s in record.findall("ConceptList/Concept/TermList/Term/String") ]
        aliases = sorted(set(aliases))
    
        entity = {'id':f'MESH:{record_ui}',
                  'name':name,
                  'aliases':aliases,
                  'definition':scope_note }
    
        ontology.append(entity)
    
    print("Loading MeSH supplementary concepts...")
    tree = ET.parse(f"{args.mesh2021_dir}/supp2021.xml")
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


