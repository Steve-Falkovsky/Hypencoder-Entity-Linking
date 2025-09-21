from transformers import set_seed, AutoTokenizer, AutoModelForTokenClassification, pipeline
from intervaltree import IntervalTree
from tqdm.auto import tqdm
from datasets import Dataset
import gc
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, DataCollatorForTokenClassification
import socket
import os
import shutil
import random
import more_itertools
import spacy
import bioc

def compute_metrics(x):
	logits, labels = x

	gold_labels = labels.reshape(-1).tolist()
	predicted_labels = logits.argmax(axis=2).reshape(-1).tolist()
	
	combined = [ (g,p) for g,p in zip(gold_labels,predicted_labels) if g != -100 ]
	gold_labels = [ g for g,p in combined ]
	predicted_labels = [ p for g,p in combined ]
	
	accuracy = accuracy_score(gold_labels, predicted_labels)
	macro_precision = precision_score(gold_labels, predicted_labels, average='macro', zero_division=0.0)
	macro_recall = recall_score(gold_labels, predicted_labels, average='macro', zero_division=0.0)
	macro_f1 = f1_score(gold_labels, predicted_labels, average='macro', zero_division=0.0)

	return {
        "accuracy": accuracy,
		"macro_precision": macro_precision,
		"macro_recall": macro_recall,
		"macro_f1": macro_f1,
	}

def tokenize_passage(tokenizer, label2id, passage, predict_variants):
    anno_tree = IntervalTree()
    for anno in passage.annotations:
        assert len(anno.locations) == 1
        start,end = anno.locations[0].offset, anno.locations[0].offset+anno.locations[0].length
        anno_tree.addi(start,end,anno.infons['concept_id'])
    
    tokenized = tokenizer(passage.text, return_offsets_mapping=True, max_length=512, truncation=True)
    
    labels = []
    prev_anno = None
    for idx,(start,end) in enumerate(tokenized['offset_mapping']):
        anno = anno_tree[start:end]
        concept_id = list(anno)[0].data if len(anno) > 0 else None
        
        if len(anno) == 0:
            labels.append( label2id['O'] )
        elif anno == prev_anno:
            labels.append( label2id['I-VARIANT'] if predict_variants and concept_id=='variant' else label2id['I-GENERAL'] )
        else:
            labels.append( label2id['B-VARIANT'] if predict_variants and concept_id=='variant' else label2id['B-GENERAL'] )
        prev_anno = anno

    return {'input_ids': tokenized['input_ids'], 'labels':labels}

def mask_negatives(x,mask_neg_rate):
    x['labels'] = [ (-100 if l == 0 and random.random() < mask_neg_rate else l) for l in x['labels']  ]
    return x

def train_mention_detector(
    train_collection,
    val_collection,
    model_name,
    learning_rate,
    batch_size,
    weight_decay,
    mask_neg_rate,
    predict_variants,
    wandb_logger=None, 
    output_dir=None
):

    if predict_variants:
        id2label = {0:'O', 1:'B-GENERAL', 2:'I-GENERAL', 3:'B-VARIANT', 4:'I-VARIANT'}
    else:
        id2label = {0:'O', 1:'B-GENERAL', 2:'I-GENERAL'}
        
    label2id = { v:k for k,v in id2label.items() }
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 512
    tokenizer.truncation = True
    
    train_dataset = [ tokenize_passage(tokenizer,label2id,passage,predict_variants) for doc in tqdm(train_collection.documents) for passage in doc.passages ]
    val_dataset = [ tokenize_passage(tokenizer,label2id,passage,predict_variants) for doc in tqdm(val_collection.documents) for passage in doc.passages ]
    
    train_dataset = Dataset.from_list(train_dataset)
    val_dataset = Dataset.from_list(val_dataset)

    max_epochs = 16

    gc.collect()
    torch.cuda.empty_cache()
    
    set_seed(42)
    random.seed(42)

    train_dataset = train_dataset.map(mask_negatives, fn_kwargs={'mask_neg_rate':0.9})
    
    model = AutoModelForTokenClassification.from_pretrained(model_name, id2label=id2label)

    unique_info = f'{socket.gethostname()}_{os.getpid()}'
    tmp_model_dir = f"tmp_mentiondetector_{unique_info}"
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,  # Stop training if no improvement after 3 evaluations
        early_stopping_threshold=0.001  # Minimum change to qualify as an improvement
    )
    
    training_args = TrainingArguments(
        output_dir=tmp_model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=max_epochs,
        weight_decay=weight_decay,
        metric_for_best_model="eval_macro_f1",
        report_to="none",
        load_best_model_at_end=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=val_dataset, 
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )
       
    trainer.train()

    eval_results = [ lh for lh in trainer.state.log_history if 'eval_macro_f1' in lh ]
    eval_results = sorted(eval_results, key=lambda x:x['eval_macro_f1'])
    metrics_to_log = eval_results[-1]

    print("Reporting metrics...")
    print(f'{metrics_to_log=}')
    if wandb_logger is not None:
        wandb_logger.log(metrics_to_log)
    
    if output_dir is not None:
        print("Saving model and tokenizer...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
    print("Cleaning up...")
    if os.path.isdir(tmp_model_dir):
        shutil.rmtree(tmp_model_dir)
    
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

def ner_to_spans(ner_result):
    spans = []
    active_start, active_end, active_variant = None,None,None
    for token in ner_result:
        label, start, end = token['entity'], token['start'], token['end']
    
        assert label in ['B-GENERAL','I-GENERAL','B-VARIANT','I-VARIANT','O']
        if label == 'B-GENERAL':
            spans.append( (active_start, active_end, active_variant) )
            active_start, active_end, active_variant = start,end,False
        elif label == 'B-VARIANT':
            spans.append( (active_start, active_end, active_variant) )
            active_start, active_end, active_variant = start,end,True
        elif label == 'I-GENERAL' and active_variant == False:
            active_end = end
        elif label == 'I-VARIANT' and active_variant == True:
            active_end = end
        elif label == 'O':
            spans.append( (active_start, active_end, active_variant) )
            active_start, active_end, active_variant = None,None,None
        else:
            active_start, active_end, active_variant = None,None,None

    spans.append( (active_start,active_end,active_variant) )

    spans = [ (start,end,is_variant) for start,end,is_variant in spans if start is not None and end is not None ]
    
    return spans

def do_ner(mention_detector_model, bioc_collection):
    # Load the model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(mention_detector_model)
    tokenizer = AutoTokenizer.from_pretrained(mention_detector_model)
    
    # Create a token classification pipeline
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device='cuda')

    bioc_passages = [ passage for doc in bioc_collection.documents for passage in doc.passages ]
    for chunk in tqdm(list(more_itertools.chunked(bioc_passages, 100))):
        texts = [ passage.text for passage in chunk ]
        ner_results = ner_pipeline(texts)

        for ner_result,passage in zip(ner_results,chunk):
            spans = ner_to_spans(ner_result)
            for start,end,is_variant in spans:
                anno = bioc.BioCAnnotation()
                anno.text = passage.text[start:end]
                loc = bioc.BioCLocation(offset=passage.offset+start, length=(end-start))
                anno.add_location(loc)
                if is_variant:
                    anno.infons['variant'] = 'True'
                
                passage.add_annotation(anno)
                
            for i,anno in enumerate(passage.annotations):
                anno.id = f"E{i+1}"
                