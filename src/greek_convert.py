# -*- coding: utf-8 -*- 
import codecs
from collections import defaultdict
from romanize import romanize
import string
from unidecode import unidecode

def tranlate_greek2latin(input_string):
    return unidecode( input_string )

def extract_token_tag(line):
    tokentags = []
    
    line = line.rstrip()
    word_tags = line.split(' ')
    for word_tag in word_tags:
        p = word_tag.rfind('/')
        #print(p)
        if p == -1: continue
        
        word = word_tag[0:p]
        tag = word_tag[p+1:]
        #print(word, tag)
        
        tokentags.append((word,tag))
    
    return tokentags

def convert_greek2latin(input, output):
    vocab = defaultdict(int)
    
    fout = codecs.open(output, 'w', 'utf-8')
    
    with codecs.open(input, 'r', 'utf-8') as f:
        for line in f:
            tokentags = extract_token_tag(line)
            #print(tokentags)
            new_tokentags = ['/'.join([tranlate_greek2latin(token),tag]) for token,tag in tokentags]
            #print(' '.join(new_tokentags))
            fout.write(' '.join(new_tokentags) + '\n')
            #break
    
            
    fout.close()

if __name__ == "__main__":
    for i in range(9):
        greek_full_training_set = '../../data/greek_treebank_perseus/greek_training_set%d.pos' %i
        output = greek_full_training_set + '.translate'
        
        convert_greek2latin(greek_full_training_set, output)
