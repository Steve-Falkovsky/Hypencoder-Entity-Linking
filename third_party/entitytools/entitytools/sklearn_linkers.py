import os
import gzip
from tqdm.auto import tqdm
from collections import defaultdict,Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import pickle
from scipy.sparse import csr_matrix
from scipy.special import expit
import numpy as np
from bioc import biocxml

from entitytools.crossencoder import make_mention_text
from entitytools.utils import collection_to_docs_passages_and_annos, add_links_to_bioc_annotations

def train_sklearn_linkers(linked_dir, crossencode_threshold, min_samples_for_clf, out_file):
    filenames = sorted(os.listdir(linked_dir))
    
    all_texts = set()
    linked_contexts = defaultdict(list)

    print("Loading and grouping crossencoder-linked texts...")
    for filename in tqdm(filenames):
        with gzip.open(os.path.join(linked_dir,filename),'rt',encoding='utf8') as f:
            collection = biocxml.load(f)

        docs_passages_annos = collection_to_docs_passages_and_annos(collection)
        for doc,passage,anno in docs_passages_annos:
            # Notably no tags are included
            window_text = make_mention_text((doc,passage,anno), only_mention=False, add_tags=False, include_title=True, window_size=100)
            all_texts.add( window_text )

            anno_links = anno.infons.get('all_links','')
            ids_and_scores = [ id_and_score.split('=') for id_and_score in anno_links.split('|') if id_and_score ]
            ids_and_scores = [ (identifier,float(score)) for identifier,score in ids_and_scores ]
            ids_and_scores = sorted(ids_and_scores, key=lambda x:x[1], reverse=True) # Do a sort just in case
            
            thresholded = [ identifier for identifier,score in ids_and_scores if score > crossencode_threshold ]
            if len(thresholded) <= 1: # Keep it simple for now and only deal with unambiguous links
                identifier = 'none' if len(thresholded) == 0 else thresholded[0]
                linked_contexts[anno.text].append( (identifier, window_text) )

    all_texts = sorted(all_texts)
    print(f"Loaded {len(all_texts)} unique texts")

    print("Building vectorizer on texts...")
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_texts)

    decisions = {}
    classifiers = {}
    others = []
    
    print(f"Training classifiers (or picking most common) for {len(linked_contexts)} mention texts...")
    for mention_text,links in tqdm(linked_contexts.items()):
        identifier_counts = Counter( identifier for identifier, context_text in links )
        identifiers_with_enough_examples = [ identifier for identifier,count in identifier_counts.items() if count >= min_samples_for_clf ]
        if len(identifiers_with_enough_examples) == 1:
            decisions[mention_text] = identifiers_with_enough_examples[0]
        elif len(identifiers_with_enough_examples) > 1:
            labels_with_texts = [ (identifier, context_text) for identifier, context_text in links if identifier in identifiers_with_enough_examples ]
    
            labels, texts = list(zip(*labels_with_texts))
    
            X_train = vectorizer.transform(texts)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                clf = LogisticRegression(random_state=0,class_weight='balanced',penalty='l2',solver='saga')
                clf.fit(X_train, labels)
    
            #classifiers[mention_text] = clf
            coefs = csr_matrix(clf.coef_)
            classifiers[mention_text] = (clf.classes_,coefs,clf.intercept_)
            #if len(classifiers) > 100:
            #    break
        else:
            others.append(mention_text)

    print("\nOutputs:")
    print(f"  {len(decisions)} mention texts with direct mapping")
    print(f"  {len(classifiers)} mention texts with classifiers")
    print(f"  {len(others)} mention texts that were ambiguous")

    print(f"\nSaving to {out_file}...")
    linkers = {'vectorizer':vectorizer, 'decisions':decisions, 'classifiers':classifiers}
    with open(out_file,'wb') as f:
        pickle.dump(linkers,f)


def apply_linkers(collection, linkers, ontology):
    ontology_by_id = { e['id']:e for e in ontology }
    ontology_id_to_idx = { e['id']:idx for idx,e in enumerate(ontology) } 
    
    print("Loading linkers...")
    with open(linkers,'rb') as f:
        linkers = pickle.load(f)

    decisions = linkers['decisions']
    classifiers = linkers['classifiers']
    vectorizer = linkers['vectorizer']

    print("Running linkers...")
    annos_to_scored_links = defaultdict(list)
    for doc in tqdm(collection.documents):
        for passage in doc.passages:
            for anno in passage.annotations:
                if anno.text in decisions:
                    concept_id = decisions[anno.text]
                    if concept_id != 'none':
                        onto_idx = ontology_id_to_idx[concept_id]
                        annos_to_scored_links[anno] = [ (1.0,onto_idx) ]
                elif anno.text in classifiers:
                    classes,coefs,intercept = classifiers[anno.text]
    
                    mention_text = make_mention_text( (doc,passage,anno) )
                    vectorized = vectorizer.transform([mention_text])
                    
                    if len(classes) == 2:
                        none_is_first = classes[0]=='none'
                        identifier = classes[1] if none_is_first else classes[0]
                        flipped_coefs = coefs if none_is_first else -coefs
                        
                        score = expit(flipped_coefs.dot(vectorized.T) + intercept).item()
                        alternatives = None

                        annos_to_scored_links[anno] = [ (score,ontology_id_to_idx[identifier]) ]
                    else:
                        scores = expit(vectorized.dot(coefs.T)+intercept).tolist()
                        scores_and_identifiers = [ (score,ontology_id_to_idx[identifier]) for score,identifier in zip(scores,classes) ]
                        

    add_links_to_bioc_annotations(collection, annos_to_scored_links, ontology)
