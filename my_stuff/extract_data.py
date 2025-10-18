import gzip
import bioc

# apparently .gz and .tar.gz are different things
# this is a tar archive
# source = "example_bioc_files.tar.gz"
# this is a gzip-compressed (.gz) single file
source = "processed_sources/bc5cdr_train.bioc.xml.gz"  # the source path is determined by pwd


with gzip.open(source) as file:
    print(f"{file.name=}")
    data = file.read().decode('utf-8')
    collection = bioc.biocxml.loads(data)
    for document in collection.documents[:5]:
        print(f"{document.id=}")
        for passage in document.passages:
            print(f"{passage.text=}")
            for anno in passage.annotations:
                print(f"{anno=}")