from lxml import etree
import os
import json
from collections import  defaultdict

__author__ = ['Kyle P. Johnson <kyle@kyle-p-johnson.com>', 'Stephen Margheim <stephen.margheim@gmail.com>']
__license__ = 'MIT License. See LICENSE.'


def get_tags():
    tag_dict = defaultdict(int)
    tag_mapping = {}
    character_tag_mapping = {}

    entire_treebank = 'latin_treebank_perseus/ldt-1.5.xml'
    with open(entire_treebank, 'r') as f:
        xml_string = f.read()
    root = etree.fromstring(xml_string)
    sentences = root.findall('sentence')

    sentences_list = []
    for sentence in sentences:  # note: sentence is Element
        words_list = sentence.findall('word')
        sentence_list = []
        # http://ilk.uvt.nl/conll/
        for x in words_list:  # note: word is class
            word = x.attrib
            #id = word['id']
            form = word['form'].lower()
            #lemma = word['lemma']
            cpostag = word['relation']  # Coarse-grained part-of-speech tag
            postag = word['postag']
            tag_dict[cpostag] += 1
            
            if cpostag not in  tag_mapping:
                tag_mapping[cpostag] = defaultdict(int)
            
            tag_mapping[cpostag][postag] += 1         
            
            #feats = '_'  # an underscore if not available
            #head = word['head']
            #deprel = word['head']
            #phead = '_'
            #pderprel = '_'
            word_tag = '/'.join([form, postag])
            sentence_list.append(word_tag)
            
            if form not in character_tag_mapping:
                character_tag_mapping[form] = defaultdict(int)

            character_tag_mapping[form][postag] += 1

        sentence_str = ' '.join(sentence_list)
        sentences_list.append(sentence_str)
    treebank_training_set = '\n\n'.join(sentences_list)

    with open('latin_training_set.pos', 'w') as f:
        f.write(treebank_training_set)
    
    with open('latin_perseus_tag_set_dis_penn.txt', 'w') as f:
        json.dump(tag_dict, f, indent=2)
    
    with open('latin_perseus_tag_set_mapping_penn2perseus.txt', 'w') as f:
        json.dump(tag_mapping, f, indent=2)
    
    ambiguity_count = 0    
    for character, tagdict in character_tag_mapping.items():
        if len(tagdict) > 1:
            ambiguity_count += 1
    print("ambiguity_count", ambiguity_count, len(character_tag_mapping), ambiguity_count*1.0/len(character_tag_mapping))

    with open('latin_perseus_character2perseustag.txt', 'w') as f:
        json.dump(character_tag_mapping, f, indent=2)

def main():
    get_tags()


if __name__ == "__main__":
    main()
