# Experiments

## Dataset Preparation

### Get MedMentions

```
bash ../scripts/setup_corpora.sh
```

### Prepare UMLS

```
# MedMentions should be compared using the same release of UMLS (2017AA-full) that was used to create it
python ../scripts/prepare_umls.py --umls /nfs/primary/umls/2017AA-full/META --out umls.json.gz
```

### (FOR DEBUGGING) Make the MedMentions corpus and UMLS ontology small

```
python ../scripts/make_debug_data.py --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --test_corpus corpora/medmentions/medmentions_test.bioc.xml.gz --ontology umls.json.gz --trim_count 10
```

## First stage (candidate generation)

### Parameter tune an n-grams candidate generation model

```
python ../scripts/train_ngrams.py --mode tune --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --ontology umls.json.gz
```

### Train and apply n-grams candidate generation model

```
python ../scripts/train_ngrams.py --mode train --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --ontology umls.json.gz --params params.json --output_dir model_ngrams
```

## Commands for running SapBERT and dense vector lookups


```
python ../scripts/make_dense_vectors.py --ontology medic.json.gz --model_name 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext' --out_vectors medic_vectors.npy

python ../scripts/make_dense_vector_lookup.py --train_corpus ncbidisease_train.bioc.xml.gz --val_corpus ncbidisease_val.bioc.xml.gz  --model_name 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext' --vectors medic_vectors.npy --output_file ncbidisease_medic_lookup.json.gz

python ../scripts/apply_ngrams.py --in_corpus ncbidisease_train.bioc.xml.gz --out_corpus test.bioc.xml.gz --candidates_lookup ncbidisease_medic_lookup.json.gz --ontology medic.json.gz

python ../scripts/evaluate.py --gold_corpus ncbidisease_train.bioc.xml.gz --pred_corpus test.bioc.xml.gz

```

## Second stage (reranker)

### Parameter tune a cross-encoder reranker

```
python ../scripts/train_crossencoder.py --mode tune --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --ontology umls.json.gz --crosstype base --ngrams_model model_ngrams
```

### Train the cross-encoder reranker

```
python ../scripts/train_crossencoder.py --mode train --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --ontology umls.json.gz  --crosstype base --ngrams_model model_ngrams --params params.json --output_dir model_crossencoder
```

### Parameter tune a parallel cross-encoder reranker

```
python ../scripts/train_crossencoder.py --mode tune --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --ontology umls.json.gz --crosstype parallel --ngrams_model model_ngrams
```

### Train the parallel cross-encoder reranker

```
python ../scripts/train_crossencoder.py --mode train --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --ontology umls.json.gz  --crosstype parallel --ngrams_model model_ngrams --params params.json --output_dir model_parallelcrossencoder
```

### Parameter tune a multi-cross-encoder reranker

```
python ../scripts/train_crossencoder.py --mode tune --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --ontology umls.json.gz --crosstype multi --ngrams_model model_ngrams
```

### Train the multi-cross-encoder reranker

```
python ../scripts/train_crossencoder.py --mode train --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --ontology umls.json.gz --crosstype multi --ngrams_model model_ngrams --params params.json --output_dir model_multicrossencoder
```

### Parameter tune a passage cross-encoder reranker

```
python ../scripts/train_crossencoder.py --mode tune --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --ontology umls.json.gz --crosstype passage --ngrams_model model_ngrams
```

### Train the passage cross-encoder reranker

```
python ../scripts/train_crossencoder.py --mode train --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --ontology umls.json.gz --crosstype passage --ngrams_model model_ngrams --params params.json --output_dir model_passagecrossencoder
```

### Parameter tune a document cross-encoder reranker

```
python ../scripts/train_crossencoder.py --mode tune --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --ontology umls.json.gz --crosstype document --ngrams_model model_ngrams
```

### Train the document cross-encoder reranker

```
python ../scripts/train_crossencoder.py --mode train --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --ontology umls.json.gz --crosstype document --ngrams_model model_ngrams --params params.json --output_dir model_documentcrossencoder
```

## Evaluating the different models

```
# Run the models on the validation dataset
# n-grams
python ../scripts/apply_ngrams.py --in_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --out_corpus results_val_ngrams.bioc.xml.gz --ngrams_model model_ngrams --ontology umls.json.gz

# SapBERT
python ../scripts/apply_ngrams.py --in_corpus ncbidisease_train.bioc.xml.gz --out_corpus test.bioc.xml.gz --candidates_lookup ncbidisease_medic_lookup.json.gz --ontology medic.json.gz

# Cross-encoder
python ../scripts/apply_reranker.py --in_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --out_corpus results_val_crossencoder.bioc.xml.gz --ngrams_model model_ngrams --ontology umls.json.gz --reranker_model model_crossencoder --crosstype base

# Parallel cross-encoder
python ../scripts/apply_reranker.py --in_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --out_corpus results_val_parallelcrossencoder.bioc.xml.gz --ngrams_model model_ngrams --ontology umls.json.gz --reranker_model model_parallelcrossencoder --crosstype parallel

# Multi cross-encoder
python ../scripts/apply_reranker.py --in_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --out_corpus results_val_multicrossencoder.bioc.xml.gz --ngrams_model model_ngrams --ontology umls.json.gz --reranker_model model_multicrossencoder --crosstype multi

# Passage cross-encoder
python ../scripts/apply_reranker.py --in_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --out_corpus results_val_passagecrossencoder.bioc.xml.gz --ngrams_model model_ngrams --ontology umls.json.gz --reranker_model model_passagecrossencoder --crosstype passage

# Document cross-encoder
python ../scripts/apply_reranker.py --in_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --out_corpus results_val_documentcrossencoder.bioc.xml.gz --ngrams_model model_ngrams --ontology umls.json.gz --reranker_model model_documentcrossencoder --crosstype document
```

```
python ../scripts/evaluate.py --gold_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --pred_corpus results_val_ngrams.bioc.xml.gz --metric_file metrics_val_ngrams.csv
python ../scripts/evaluate.py --gold_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --pred_corpus results_val_crossencoder.bioc.xml.gz --metric_file metrics_val_crossencoder.csv
python ../scripts/evaluate.py --gold_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --pred_corpus results_val_parallelcrossencoder.bioc.xml.gz --metric_file metrics_val_parallelcrossencoder.csv
python ../scripts/evaluate.py --gold_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --pred_corpus results_val_multicrossencoder.bioc.xml.gz --metric_file metrics_val_multicrossencoder.csv
python ../scripts/evaluate.py --gold_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --pred_corpus results_val_passagecrossencoder.bioc.xml.gz --metric_file metrics_val_passagecrossencoder.csv
python ../scripts/evaluate.py --gold_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --pred_corpus results_val_documentcrossencoder.bioc.xml.gz --metric_file metrics_val_documentcrossencoder.csv
```

## Mention Detection Setup

### Parameter tune a mention detector

```
python ../scripts/train_mention_detector.py --mode tune --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz
```

### Train the mention detector

```
python ../scripts/train_mention_detector.py --mode train --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --params params.json --output_dir model_mention_detector
```


## Sampling mentions from PubMed

### Apply mention detection to PubMed data

```
bash ../scripts/generate_pubmed_mention_detect_joblist.sh model_mention_detector pubmed_with_mentions mention_detector.joblist
```

Run the generated job-list `mention_detector.joblist` (potentially in parallel)
```
#bash /home/jakelever/entitytools/scripts/download_pubmed_and_mention_detect.sh ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed25n1274.xml.gz /home/jakelever/entitytools/experiments/model_mention_detector /home/jakelever/entitytools/experiments/pubmed_with_mentions
```

### Sample mentions (100 of each one) from PubMed files

```
mkdir -p sampled

# This samples only mentions that occur in MedMentions
#python ../scripts/sample_mentions.py --mode corpora --train_corpus corpora/medmentions/medmentions_train.bioc.xml.gz --val_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --input_dir pubmed_with_mentions --out_dir sampled

# This samples everything (but filters out anything that occurs less than 100 times)
python ../scripts/sample_mentions.py --mode all --input_dir pubmed_with_mentions --sample_size 100 --require_complete --out_dir sampled --out_mentions sampled_mentions.txt.gz
```
### Apply the n-grams (first-stage) to make a lookup for the sampled mentions

```
python ../scripts/make_ngrams_lookup.py --mention_texts sampled_mentions.txt.gz --ngrams_model model_ngrams --top_k 5 --output_file sampled_lookup.json.gz
```

### Apply crossencoder to sampled PubMed data for MedMentions mentions

```
mkdir -p linked
python ../scripts/apply_reranker.py --in_corpus sampled/pubmed25n1274.bioc.xml.gz --candidates_lookup sampled_lookup.json.gz --reranker_model model_crossencoder --ontology umls.json.gz --out_corpus linked/pubmed25n1274.bioc.xml.gz
```

## Pick the most commonly linked entity for each mention

```
# Make a lookup that uses the most common linked entities
# This probably needs a lot of PubMed to be sampled to work well
python ../scripts/make_frequent_lookup.py --in_dir linked --output_file sampled_frequent_lookup.json.gz --ontology umls.json.gz --lowercase True
python ../scripts/apply_ngrams.py --in_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --out_corpus results_val_frequentlinked.bioc.xml.gz --candidates_lookup sampled_frequent_lookup.json.gz --ontology umls.json.gz
python ../scripts/evaluate.py --gold_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --pred_corpus results_val_frequentlinked.bioc.xml.gz
```

## Scikit-learn-based linkers

```
python ../scripts/train_sklearn_linkers.py --linked_dir linked --crossencode_threshold 0.7 --min_samples_for_clf 10 --out_file sklearn_linkers.pickle
python ../scripts/apply_sklearn_linkers.py --in_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --out_corpus results_val_sklearn_linkers.bioc.xml.gz --linkers sklearn_linkers.pickle --ontology umls.json.gz
python ../scripts/evaluate.py --gold_corpus corpora/medmentions/medmentions_val.bioc.xml.gz --pred_corpus results_val_sklearn_linkers.bioc.xml.gz

```