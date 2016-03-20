"""An example using CLTK taggers."""

from cltk.corpus.utils.importer import CorpusImporter
from cltk.tag import pos

def pos_tagger_example_latin():
    corpus_importer = CorpusImporter('latin')
    corpus_importer.import_corpus('latin_models_cltk')

    tagger = pos.POSTag('latin')
    pos_tags = tagger.tag_ngram_123_backoff('Gallia est omnis divisa in partes tres')

    print(pos_tags)
    
if __name__ == "__main__":
    pos_tagger_example_latin()

    
