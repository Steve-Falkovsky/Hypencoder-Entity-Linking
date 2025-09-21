from collections import defaultdict
import json
import gzip
import argparse

def load_umls(umls_dir):
	names = {}
	aliases = defaultdict(set)
	definitions = defaultdict(list)
	semantic_types = defaultdict(set)
	broader_concepts = defaultdict(set)

	# MRCONSO.RRF contains concepts and their synonyms
	# Columns in MRCONSO.RRF explained at https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/
	with open(f'{umls_dir}/MRCONSO.RRF',encoding='utf8') as f:
		for line in f:
			split = line.rstrip('\n').split('|')
			cui = split[0]
			language = split[1]
			if language != 'ENG':
				continue

			term_type = split[12]
			entity_name = split[14]

			if cui not in names:
				names[cui] = entity_name

			# Abbreviations for term type defined at https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/abbreviations.html#mrdoc_TTY
			# These are a short list of term types for synonyms, acronyms and etc that should be exact matches to the term (and not related concepts)
			if term_type in ['PT','BN','MH','SY','SYN','SYGB','ACR','PN']:
				aliases[cui].add(entity_name)

	print("Dealing with two concepts (C4300518 and C4300640) used in MedMentions without English info")
	names['C4300518'] = 'Disability-adjusted life years'
	names['C4300640'] = 'Horticulture'

	# MRDEF.RRF contains definitions
	# Columns in MRDEF.RRF explained at https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.definitions_file_mrdef_rrf/
	with open(f'{umls_dir}/MRDEF.RRF',encoding='utf8') as f:
		for line in f:
			split = line.rstrip('\n').split('|')
			cui = split[0]
			source = split[4]
			definition = split[5]

			definitions[cui].append( (source,definition) )
			
	# MRSTY.RRF contains the semantic types of entities
	# Columns in MRSTY.RRF explained at https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.Tf/
	with open(f'{umls_dir}/MRSTY.RRF',encoding='utf8') as f:
		for line in f:
			split = line.rstrip('\n').split('|')
			cui = split[0]
			semantic_type = split[3]
			semantic_types[cui].add( semantic_type )
			
	# MRREL.RRF contains the relationships between entities
	# Columns in MRREL.RRF explained at https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.related_concepts_file_mrrel_rrf/
	with open(f'{umls_dir}/MRREL.RRF',encoding='utf8') as f:
		for line in f:
			split = line.rstrip('\n').split('|')
			cui1 = split[0]
			rel_type = split[3]
			cui2 = split[4]

			# Abbreviation for relation type defined at https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/abbreviations.html#mrdoc_REL
			if rel_type in ['RB','PAR']: #cui2 is a broader/parent concept of cui1
				broader_concepts[cui1].add(cui2)

	entities = []
	for cui in names:
		entity = {'id':cui,
				'name':names[cui],
				'aliases':sorted(aliases[cui]),
				'definitions':definitions[cui],
				'semantic_types':sorted(semantic_types[cui]),
				'broader_concepts': [ names[cui2] for cui2 in sorted(broader_concepts[cui]) if cui2 in names ] }
		entities.append(entity)

	return entities
	
	
def main():
	parser = argparse.ArgumentParser(description='Load UMLS and save it to a GZipped JSON file')
	parser.add_argument('--umls_dir',required=True,type=str,help='Directory with UMLS files')
	parser.add_argument('--out',required=True,type=str,help='Output Gzipped JSON file')
	args = parser.parse_args()
	
	print("Loading UMLS...")
	umls = load_umls(args.umls_dir)
	print(f"Loaded {len(umls)} entities")
	
	with gzip.open(args.out,'wt',encoding='utf8') as f:
		json.dump(umls,f)

	print("Done")
	
if __name__ == '__main__':
	main()
