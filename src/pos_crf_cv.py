"""An evaluation of CLTK taggers."""

from nltk.corpus.reader import TaggedCorpusReader
from nltk.tag import CRFTagger
from nltk.tokenize import wordpunct_tokenize
import math
import os
import pandas as pd
import random
from statistics import mean
from statistics import stdev
import sys

def cltk_pos_cv(full_training_set, local_dir_rel):
    print("full_training_set", full_training_set)

    crf_accuracies = []
    
    with open(full_training_set) as f:
        training_set_string = f.read()

    pos_set = training_set_string.split('\n\n')  # mk into a list

    sentence_count = len(pos_set)  # 3473
    tenth = math.ceil(int(sentence_count) / int(10))

    random.seed(0)
    random.shuffle(pos_set)

    def chunks(l, n):
        """Yield successive n-sized chunks from l.
        http://stackoverflow.com/a/312464
        """
        for i in range(0, len(l), n):
            yield l[i:i+n]

    # a list of 10 lists
    ten_parts = list(chunks(pos_set, tenth))  # a list of 10 lists with ~347 sentences each

    #for counter in list(range(10)):
    for counter, part in list(enumerate(ten_parts)):
        # map test list to part of given loop
        test_set = ten_parts[counter]  # or: test_set = part
        
        # filter out this loop's test index
        training_set_lists = [x for x in ten_parts if x is not ten_parts[counter]]
        
        # next concatenate the list together into 1 file ( http://stackoverflow.com/a/952952 )
        training_set = [item for sublist in training_set_lists for item in sublist]
            
        # save shuffled tests to file (as NLTK trainers expect)
        #local_dir_rel = '~/cltk_data/user_data'
        local_dir = os.path.expanduser(local_dir_rel)
        if not os.path.isdir(local_dir):
            os.makedirs(local_dir)

        test_path = os.path.join(local_dir, 'test.pos')
        with open(test_path, 'w') as f:
            f.write('\n\n'.join(test_set))

        train_path = os.path.join(local_dir, 'train.pos')
        with open(train_path, 'w') as f:
            f.write('\n\n'.join(training_set))

        # read POS corpora
        print("local_dir", local_dir)
        train_reader = TaggedCorpusReader(local_dir, 'train.pos')
        train_sents = train_reader.tagged_sents()

        test_reader = TaggedCorpusReader(local_dir, 'test.pos')
        test_sents = test_reader.tagged_sents()
        
        print('Loop #' + str(counter))
        # make crf tagger
        crf_tagger = CRFTagger()
        crf_tagger.train(train_sents, 'model.crf.tagger')
        
        # evaluate crf tagger
        crf_accuracy = None
        crf_accuracy = crf_tagger.evaluate(test_sents)
        crf_accuracies.append(crf_accuracy)
        print('crf:', crf_accuracy)

        #if counter> 0: break
        
    final_accuracies_list = []
    mean_accuracy_crf = mean(crf_accuracies)
    standard_deviation_crf = stdev(crf_accuracies)
    uni = {'crf': {'mean': mean_accuracy_crf, 'sd': standard_deviation_crf}}
    final_accuracies_list.append(uni)

    final_dict = {}
    for x in final_accuracies_list:
        final_dict.update(x)
    
    return final_dict

if __name__ == "__main__":
    #latin_full_training_set = '/media/wencan/Private/project/gsoc2016/data/latin_treebank_perseus/latin_training_set.pos'
    #greek_full_training_set = '/media/wencan/Private/project/gsoc2016/data/greek_treebank_perseus/greek_training_set.pos'
    #local_dir_rel = '~/cltk_data/user_data'

    full_training_set = sys.argv[1]
    local_dir_rel = sys.argv[2]
    
    final_dict = cltk_pos_cv(full_training_set, local_dir_rel)

    df = pd.DataFrame(final_dict)
    print(df)
    

    

    
