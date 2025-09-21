import random
import os
from collections import defaultdict, Counter
from tqdm.auto import tqdm
import gzip
from bioc import biocxml

from entitytools.utils import get_process_memory

class ReservoirSampler:
    def __init__(self, k):
        """
        Initializes the reservoir sampler with a size of k.
        :param k: Number of items to sample.
        """
        self.k = k
        self.reservoir = []
        self.n = 0  # Count of total items encountered so far

    def process(self, item):
        """
        Processes an item from the stream.
        :param item: The new item from the stream.
        """
        self.n += 1
        
        if len(self.reservoir) < self.k:
            # If reservoir is not full, simply append the new item
            self.reservoir.append(item)
        else:
            # Replace an item in the reservoir with decreasing probability
            idx = random.randint(0, self.n - 1)
            if idx < self.k:
                self.reservoir[idx] = item

    def get_samples(self):
        """
        Returns the current set of sampled items.
        :return: The reservoir with k sampled items.
        """
        return self.reservoir


def sample_annotations_from_bioc_dir(input_dir, sample_size, require_complete, out_dir, out_mentions=None, annotation_text_filter=None):
    
    assert os.path.isdir(input_dir), "input_dir must be a directory"
    bioc_filenames = [ x for x in sorted(os.listdir(input_dir)) if x.endswith('.bioc.xml.gz') ]

    print(f"Found {len(bioc_filenames)} BioC XML files to process")

    print("Processing files and sampling mentions to keep...")
    selected = defaultdict(lambda : ReservoirSampler(sample_size))
    pbar = tqdm(bioc_filenames)
    for file_idx,bioc_filename in enumerate(pbar):
        with gzip.open(f'{input_dir}/{bioc_filename}','rt',encoding='utf8') as f:
            collection = biocxml.load(f)
        
        for doc_idx,doc in enumerate(collection.documents):
            for passage_idx,passage in enumerate(doc.passages):
                passage.infons['doc_id'] = doc.id
                for anno_idx,anno in enumerate(passage.annotations):
                    if annotation_text_filter is None or anno.text in annotation_text_filter:
                        selected[anno.text].process( (file_idx,doc_idx,passage_idx,anno_idx) )
    
        process_mb = get_process_memory()
        pbar.set_description(f'{process_mb:.1f}MB')

    print(f"Identified {len(selected)} unique mentions")
    count_with_full = sum( 1 for mention_text,reservoir_sampler in selected.items() if len(reservoir_sampler.reservoir) >= sample_size )
    print(f"% with full ({sample_size}) sample set = {count_with_full}/{len(selected)} = {100*count_with_full/len(selected):.1f}%")

    if require_complete:
        print(f"Reducing mentions to those with complete {sample_size} samples...")
        selected = { mention_text:reservoir_sampler for mention_text,reservoir_sampler in selected.items() if len(reservoir_sampler.reservoir) >= sample_size }
        print(f"Filtered to {len(selected)} unique mentions")
              
    print("Indexing choices...")
    selected_mentions_by_file = defaultdict(list)
    for mention_text in selected:
        sampled = selected[mention_text].reservoir
        for file_idx,doc_idx,passage_idx,anno_idx in sampled:
            selected_mentions_by_file[file_idx].append( (doc_idx,passage_idx,anno_idx) )
    
    print("Reprocessing PubMed files and saving out mentions...")
    for file_idx,bioc_filename in enumerate(tqdm(bioc_filenames)):
        with gzip.open(f'{input_dir}/{bioc_filename}','rt',encoding='utf8') as f:
            collection = biocxml.load(f)

        selected_in_this_file = set(selected_mentions_by_file[file_idx])

        # Trim down to minimal set of documents & passages that contain the needed annotations
        for doc_idx,doc in enumerate(collection.documents):
            for passage_idx,passage in enumerate(doc.passages):
                passage.annotations = [ anno for anno_idx,anno in enumerate(passage.annotations) if (doc_idx,passage_idx,anno_idx) in selected_in_this_file ]
            doc.passages = [ passage for passage in doc.passages if passage.annotations ]
        collection.documents = [ doc for doc in collection.documents if doc.passages ]
            
        with gzip.open(f'{out_dir}/{bioc_filename}','wt',encoding='utf8') as f:
            biocxml.dump(collection, f)

    if out_mentions:
        with gzip.open(out_mentions,'wt',encoding='utf8') as f:
            mention_texts = sorted(selected.keys())
            for mention_text in mention_texts:
                f.write(f"{mention_text}\n")

def make_frequent_lookup(collections, ontology, lowercase, score_threshold=None, top_k=None):
    ontology_id_to_idx = { e['id']:idx for idx,e in enumerate(ontology) }

    anno_link_counts = defaultdict(Counter)
    for collection in collections:
        for doc in collection.documents:
            for passage in doc.passages:
                for anno in passage.annotations:
                    anno_text = anno.text.lower() if lowercase else anno.text
                    if 'concept_id' in anno.infons and anno.infons['concept_id'] != 'none':
                        concept_id = anno.infons['concept_id']
                        score = float(anno.infons['concept_score'])
                        onto_idx = ontology_id_to_idx[concept_id]

                        if score_threshold is None or score > score_threshold:
                            anno_link_counts[anno_text][onto_idx] += 1

    lookup = {}
    for anno_text,counts in anno_link_counts.items():
        most_common = [ onto_idx for onto_idx,count in counts.most_common() ]
        if top_k is not None:
            most_common = most_common[:top_k]
        lookup[anno_text] = most_common
    return lookup