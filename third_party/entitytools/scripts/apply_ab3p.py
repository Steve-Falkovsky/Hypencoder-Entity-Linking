import gzip
from bioc import biocxml
import json
import argparse
import os
import subprocess
from tqdm.auto import tqdm

def get_abbrevs_from_ab3b_output(ab3p_output):
    abbrevs = []
    for line in ab3p_output.splitlines():
        if line.startswith('  ') and line.count('|') == 2:
            short_form, long_form, score = line.split('|')
            score = float(score)
            abbrevs.append( (short_form.lstrip(), long_form, score) )
    return abbrevs


def main():
    parser = argparse.ArgumentParser('Run the acronym extraction method on a BioC XML file')
    parser.add_argument('--in_corpus',required=True,type=str,help='Gzipped BioC XML file')
    parser.add_argument('--ab3p_dir',required=True,type=str,help='Directory containing Ab3P')
    parser.add_argument('--output',required=True,type=str,help='Gzipped JSON file with abbreviations')
    args = parser.parse_args()

    output_filename = os.path.abspath(args.output)

    assert os.path.isdir(args.ab3p_dir), "--ab3p_dir must be a directory with a compiled version of Ab3P"
    assert os.access(f"{args.ab3p_dir}/identify_abbr", os.X_OK), "--ab3p_dir must be a directory with a compiled version of Ab3P"

    with gzip.open(args.in_corpus,'rt',encoding='utf8') as f:
        collection = biocxml.load(f)
    
    print(f"Loaded {len(collection.documents)} documents")

    os.chdir(args.ab3p_dir)

    print("Running Ab3P...")
    abbrevs = {}
    for doc in tqdm(collection.documents):
        assert len(doc.passages) == 1
        passage = doc.passages[0]
        
        with open('tmp_input.txt','w') as f:
            f.write(passage.text)

        result = subprocess.run(['./identify_abbr', 'tmp_input.txt'], capture_output=True, text=True)

        abbrevs[doc.id] = get_abbrevs_from_ab3b_output(result.stdout)

    print("Saving...")
    with gzip.open(output_filename,'wt') as f:
        json.dump(abbrevs,f,indent=2)

    print("Done.")

if __name__ == '__main__':
    main()
    
