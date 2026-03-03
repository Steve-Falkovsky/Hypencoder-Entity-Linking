
import bioc
from bioc import biocxml, pubtator
import argparse
from intervaltree import IntervalTree
import gzip

from entitytools.file_formats import pubtator_to_bioc, save_bioc_docs

def count_annotations(documents):
	annos = [ anno for document in documents for passage in document.passages for anno in passage.annotations ]
	return len(annos)

def remove_docs_without_annotations(docs):
	trimmed = [ doc for doc in docs if len( [ anno for passage in doc.passages for anno in passage.annotations ]) > 0 ]
	return trimmed
	

def main():
	parser = argparse.ArgumentParser(description='Convert the MedMentions corpus into BioC XML format')
	parser.add_argument('--medmentions_dir',required=True,type=str,help='Directory with source MedMentions files')
	parser.add_argument('--out_train',required=True,type=str,help='Output Gzipped BioC XML for training data')
	parser.add_argument('--out_val',required=True,type=str,help='Output Gzipped BioC XML for validation data')
	parser.add_argument('--out_test',required=True,type=str,help='Output Gzipped BioC XML for test data')
	parser.add_argument('--clean',action='store_true',help='Whether to remove overlapping annotations and duplicate annotations')
	args = parser.parse_args()
	
	with open(f"{args.medmentions_dir}/corpus_pubtator_pmids_trng.txt") as f:
		train_pmids = set( line.strip() for line in f )
	with open(f"{args.medmentions_dir}/corpus_pubtator_pmids_dev.txt") as f:
		val_pmids = set( line.strip() for line in f )
	with open(f"{args.medmentions_dir}/corpus_pubtator_pmids_test.txt") as f:
		test_pmids = set( line.strip() for line in f )
		
	print(f"{len(train_pmids)=} {len(val_pmids)=} {len(test_pmids)=}")
	
	train_docs, val_docs, test_docs = [], [], []
	with gzip.open(f"{args.medmentions_dir}/corpus_pubtator.txt.gz", 'rt',encoding='utf8') as fp:
		pubtator_docs = pubtator.load(fp)
	
	for pubtator_doc in pubtator_docs:
		bioc_doc = pubtator_to_bioc(pubtator_doc)
		
		if bioc_doc.id in train_pmids:
			train_docs.append(bioc_doc)
		elif bioc_doc.id in val_pmids:
			val_docs.append(bioc_doc)
		elif bioc_doc.id in test_pmids:
			test_docs.append(bioc_doc)
		else:
			raise RuntimeError(f"ID not assigned to a split. {bioc_doc.id=}")
			
	
	if args.clean:
		print("Removing any duplicate annotations")

		print(f"{count_annotations(train_docs+val_docs+test_docs)=}")
		for doc in (train_docs+val_docs+test_docs):
			for passage in doc.passages:
				annos = set()
				for anno in passage.annotations:
					assert len(anno.locations) == 1, f"{anno=}"
					start = anno.locations[0].offset - passage.offset
					end = start + anno.locations[0].length

					annos.add( (anno.locations[0].offset, anno.locations[0].length, anno.text, anno.infons['concept_id']))
				
				deduplicated = []
				for offset, length, text, concept_id in sorted(annos):
					bioc_anno = bioc.BioCAnnotation()
					bioc_anno.text = text
					bioc_loc = bioc.BioCLocation(offset, length)
					bioc_anno.add_location(bioc_loc)
					bioc_anno.infons['concept_id'] = concept_id
					deduplicated.append(bioc_anno)

				passage.annotations = deduplicated
				
		print(f"{count_annotations(train_docs+val_docs+test_docs)=}")

		print("Removing any overlapping annotations")

		print(f"{count_annotations(train_docs+val_docs+test_docs)=}")
		for doc in train_docs+val_docs+test_docs:
			for passage in doc.passages:
				anno_tree = IntervalTree()
				for anno in passage.annotations:
					assert len(anno.locations) == 1, f"{anno=}"
					start = anno.locations[0].offset - passage.offset
					end = start + anno.locations[0].length
					entity_id = anno.infons['concept_id']
					anno_tree.addi(start, end, entity_id)

				filtered_annotations = []
				for anno in passage.annotations:
					assert len(anno.locations) == 1, f"{anno=}"
					start = anno.locations[0].offset - passage.offset
					end = start + anno.locations[0].length

					if len(anno_tree[start:end]) == 1:
						filtered_annotations.append(anno)
				
				passage.annotations=filtered_annotations
		
		print(f"{count_annotations(train_docs+val_docs+test_docs)=}")

		print("Removing documents without any annotations")
		print(f"{len(train_docs)=} {len(val_docs)=} {len(test_docs)=}")
		train_docs = remove_docs_without_annotations(train_docs)
		val_docs = remove_docs_without_annotations(val_docs)
		test_docs = remove_docs_without_annotations(test_docs)
		print(f"{len(train_docs)=} {len(val_docs)=} {len(test_docs)=}")
		
	print("Do some checks on the combined set of processed documents")
	for doc in train_docs+val_docs+test_docs:
		for passage in doc.passages:
			assert len(passage.text) > 0
			for anno in passage.annotations:
				assert len(anno.locations) == 1
				assert anno.locations[0].length > 0

				start = anno.locations[0].offset - passage.offset
				end = start + anno.locations[0].length

				assert start >= 0 and end <= len(passage.text)

				assert passage.text[start:end] == anno.text
				assert 'concept_id' in anno.infons

	print("Saving documents")
	save_bioc_docs(train_docs, args.out_train)
	save_bioc_docs(val_docs, args.out_val)
	save_bioc_docs(test_docs, args.out_test)

	print(f"{len(train_docs)=} {len(val_docs)=} {len(test_docs)=}")
	print("Done")
	
if __name__ == '__main__':
	main()
