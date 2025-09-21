import gzip
import xml.etree.cElementTree as etree
import bioc
from bioc import biocxml

def pubmed_to_bioc(pubmed_filename):
    """
    Function that transforms a PubMed article to BioC
    :param pubmed_filename: the name of the PubMed article.
    :return: the BioC document.
    """
    docs = []
    with gzip.open(pubmed_filename,'rt',encoding='utf8') as f:
        for event, elem in etree.iterparse(f, events=("start", "end", "start-ns", "end-ns")):
    
            # Iterate through the XML file until a <PubmedArticle> tag is closed (which we then process below)
            if event == "end" and elem.tag == "PubmedArticle":
    
                pmid = int(elem.find("./MedlineCitation/PMID").text)
                title = elem.find("./MedlineCitation/Article/ArticleTitle").text
    
                abstract_elems = elem.findall("./MedlineCitation/Article/Abstract/AbstractText")
                abstract = [ "".join(e.itertext()) for e in abstract_elems ]

                if not title:
                    continue
    
                doc = bioc.BioCDocument()
                doc.id = pmid
                doc.infons['title'] = title
    
                title_passage = bioc.BioCPassage()
                title_passage.text = title
                title_passage.offset = 0
                doc.add_passage(title_passage)
    
                offset = len(title_passage.text)
                for section in abstract:
                    if len(section) > 0:
                        abstract_passage = bioc.BioCPassage()
                        abstract_passage.text = section
                        abstract_passage.offset = offset
                        doc.add_passage(abstract_passage)
                        offset += len(abstract_passage.text)
    
                docs.append(doc)
    
                elem.clear()

    return bioc.BioCCollection.of_documents(*docs)

def pubtator_to_bioc(doc):
    """
    Function that transforms a PubTator document into BioC.
    :param doc: the PubTator document.
    :return: the BioC document.
    """
    bioc_doc = bioc.BioCDocument()
    bioc_doc.id = doc.pmid
    bioc_passage = bioc.BioCPassage()
    bioc_passage.text = doc.text
    bioc_passage.offset = 0
    bioc_doc.add_passage(bioc_passage)

    title = doc.text.split('\n')[0]
    bioc_doc.infons['title'] = title

    for a in doc.annotations:
        bioc_anno = bioc.BioCAnnotation()
        bioc_anno.infons['concept_id'] = a.id
        bioc_anno.text = a.text
        bioc_loc = bioc.BioCLocation(a.start,a.end-a.start)
        bioc_anno.add_location(bioc_loc)
        bioc_passage.add_annotation(bioc_anno)

    return bioc_doc

def save_bioc_docs(docs, filename):
    """
    Saves a set of BioC documents.
    :param docs: the set of documents.
    :param filename: the name of the file on which to save the BioC docs.
    """
    collection = bioc.BioCCollection.of_documents(*docs)
    with gzip.open(filename, 'wt', encoding='utf8') as f:
        biocxml.dump(collection, f)
