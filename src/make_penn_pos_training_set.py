from lxml import etree
import os
from collections import  defaultdict
import json

__author__ = ['Kyle P. Johnson <kyle@kyle-p-johnson.com>', 'Stephen Margheim <stephen.margheim@gmail.com>']
__license__ = 'MIT License. See LICENSE.'


def get_tags():
    tag_dict = defaultdict(int)
    character_tag_mapping = {}

    entire_treebank = 'latin_treebank_perseus/ldt-1.5.xml'
    with open(entire_treebank, 'r') as f:
        xml_string = f.read()
    root = etree.fromstring(xml_string)
    sentences = root.findall('sentence')
    
    character_dict = defaultdict(int)

    sentences_list = []
    for sentence in sentences:  # note: sentence is Element
        words_list = sentence.findall('word')
        sentence_list = []
        # http://ilk.uvt.nl/conll/
        for x in words_list:  # note: word is class
            word = x.attrib
            #id = word['id']
            form = word['form']
            #lemma = word['lemma']
            cpostag = word['relation']  # Coarse-grained part-of-speech tag
            postag = word['postag']
            #feats = '_'  # an underscore if not available
            #head = word['head']
            #deprel = word['head']
            #phead = '_'
            #pderprel = '_'
            
            #cpostag = cpostag.split('_')[0]
            tag_dict[cpostag] += 1

            word_tag = '/'.join([form, cpostag])
            sentence_list.append(word_tag)
            
            if form not in character_tag_mapping:
                character_tag_mapping[form] = defaultdict(int)

            character_tag_mapping[form][cpostag] += 1
            character_dict[form] += 1

        sentence_str = ' '.join(sentence_list)
        sentences_list.append(sentence_str)
    treebank_training_set = '\n\n'.join(sentences_list)

    with open('penn_pos_training_set_reduce.pos', 'w') as f:
        f.write(treebank_training_set)
    
    with open('latin_perseus_tag_set_dis_penn_reduce.txt', 'w') as f:
        json.dump(tag_dict, f, indent=2)
    
    total = 0
    oracle_incorrect = 0
    characters = sorted(character_dict, key=character_dict.get, reverse=True)
    character_tag_mapping_list = []
    for char in characters:
        character_tag_mapping_list.append({char:character_tag_mapping[char]})
        total += character_dict[char]
        
        tags = sorted(character_tag_mapping[char], key=character_tag_mapping[char].get, reverse=True)
        oracle_incorrect += character_dict[char] - character_tag_mapping[char][tags[0]]
    print("oracle_incorrect", oracle_incorrect, "total", total, oracle_incorrect*1.0/total)
    
    with open('latin_perseus_character2penn.txt', 'w') as f:
        json.dump(character_tag_mapping_list, f, indent=2)

def main():
    get_tags()


if __name__ == "__main__":
    main()
