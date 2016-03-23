#python /media/wencan/Private/project/gsoc2016/cltk_pos/src/pos_crf_cv_quick.py /media/wencan/Private/project/gsoc2016/data/greek_treebank_perseus/greek_training_set.pos ~/cltk_data/user_data/greek_crf_quick 1

for i in $(seq 0 9)
do
    python /media/wencan/Private/project/gsoc2016/cltk_pos/src/pos_crf_cv_quick.py /media/wencan/Private/project/gsoc2016/data/latin_treebank_perseus/latin_training_set.pos ~/cltk_data/user_data/greek_crf_quick 2 $i &
done
