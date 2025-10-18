import gzip
import random
import bioc
import json

# apparently .gz and .tar.gz are different things
# this is a tar archive
# source = "example_bioc_files.tar.gz"
# this is a gzip-compressed (.gz) single file
source = "processed_sources/bc5cdr_train.bioc.xml.gz"  # the source path is determined by pwd


with gzip.open(source) as file:
    print(f"{file.name=}")
    data = file.read().decode('utf-8')
    collection = bioc.biocxml.loads(data)
    for document in collection.documents[:5]: # get first 5 documents
        print(f"{document.id=}")
        for passage in document.passages:
            print(f"{passage.text=}")
            for anno in passage.annotations:
                print(f"{anno=}")
                
                
                
mesh = "processed_sources/mesh2015.json.gz"

with gzip.open(mesh, 'rt', encoding='utf-8') as file:
    print(f"{file.name=}")
    data = json.load(file)

    if isinstance(data, list):  
        print(f"Number of items: {len(data)}")
        
        # Print random items for inspection
        for item in random.sample(data, 15):
            print(json.dumps(item, indent=2))
