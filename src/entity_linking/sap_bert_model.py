from .utils import DATA_PROCESSED, extract_first_n_docs, get_entry_by_id, read_first_n_from_json_gz

# Load model directly
from transformers import AutoTokenizer, AutoModel

model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

docs = extract_first_n_docs(DATA_PROCESSED / "bc5cdr_train.bioc.xml.gz", 5)

# get only the text of the passages
texts = [passage['text'] for doc in docs for passage in doc['passages']]

# get the annotations (inside each passage)
annotations = [passage['annotations'] for doc in docs for passage in doc['passages']]

print("\n\n\n")

# print annotations from first n passages
n = 3
for i in range(n):
    print(f"Annotations in passage {i+1}: {annotations[i]}\n")

# Match BC5CDR to Mesh IDs
# read all mesh data, then use the id from the BC5CDR entry to get the corresponding mesh term
mesh_data = read_first_n_from_json_gz(DATA_PROCESSED / "mesh2015.json.gz", None)


# take annotations from first n passages and match to mesh terms
for i in range(2):
    print(f"Matching annotations for passage {i+1}:\n")
    for j, anno in enumerate(annotations[i]):
        mesh_id = anno['infons']['concept_id']
        mesh_entry = get_entry_by_id(mesh_data, mesh_id)
        print(
            f"Annotation {j+1}: MeSH ID: {mesh_id}, MeSH name: {mesh_entry['name']} \n"
            f"MeSH Aliases: {mesh_entry['aliases']}\n\n"
            f"definition: {mesh_entry['definition']}\n"
            f"{'-'*50}\n"
        )


# Tokenize the texts
# inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

#print(f"first input: {inputs['input_ids'][0]}\n")