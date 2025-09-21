from tqdm.auto import tqdm
import bioc
from intervaltree import IntervalTree


def mark_sentences(collection):
    """
    Marks the sentences within a collection. Sentences are marked within the collection.
    :param collection: the collection of documents.
    """
    import spacy
    nlp = spacy.load("en_core_web_sm")

    true_start = -1
    start = -1
    end = -1
    for doc in tqdm(collection.documents):
        for passage in doc.passages:
            if len(passage.annotations) > 0:
                i = 0
                current_anno = passage.annotations[i]
                start_offset = current_anno.locations[0].offset
                end_offset = start_offset + current_anno.locations[0].length
            else:
                start_offset = 10000000
                end_offset = 10000000
            parsed = nlp(passage.text)
            true_start = -1
            for sent in parsed.sents:
                start = sent[0].idx
                end = sent[-1].idx + len(sent[-1].text)
                while (end_offset - passage.offset) <= end and (i + 1) < len(passage.annotations):
                    i += 1
                    current_anno = passage.annotations[i]
                    start_offset = current_anno.locations[0].offset
                    end_offset = start_offset + current_anno.locations[0].length

                if (start_offset - passage.offset) > end or (end_offset - passage.offset) < end:
                    start = true_start if true_start != -1 else start
                    sentence = bioc.BioCSentence()
                    sentence.offset = passage.offset + start
                    sentence.infons['length'] = (end - start)
                    passage.add_sentence(sentence)
                    true_start = -1
                elif true_start == -1:
                    true_start = start
            if true_start != -1:
                sentence = bioc.BioCSentence()
                sentence.offset = passage.offset + true_start
                sentence.infons['length'] = len(passage.text) - true_start
                passage.add_sentence(sentence)
                true_start = -1


def link_annotations_to_sentences(collection):
    """
    Links annotations to sentences within a collection. Annotations are stored within the collection.
    :param collection: the collection of documents.
    """
    for doc in tqdm(collection.documents):
        for passage in doc.passages:

            anno_tree = IntervalTree()
            for anno in passage.annotations:
                assert len(anno.locations) == 1
                loc = anno.locations[0]
                anno_tree.addi(loc.offset, loc.offset + loc.length, anno)

            for sentence in passage.sentences:
                sentence_start = sentence.offset
                sentence_end = sentence.offset + int(sentence.infons['length'])

                sentence.annotations = [interval.data for interval in anno_tree[sentence_start:sentence_end] if
                                        interval.begin >= sentence_start and interval.end <= sentence_end]


def get_sentence_annotations(passage, sentence):
    """
    Obtains the annotations within a sentence.
    :param passage: the complete passage.
    :param sentence: a sentence within the passage.
    :return: the annotations for the sentence.
    """
    sentence_start = sentence.offset
    sentence_end = sentence.offset + int(sentence.infons['length'])

    annotations = [anno for anno in passage.annotations if
                   anno.total_span.offset >= sentence_start and anno.total_span.end <= sentence_end]

    return annotations
