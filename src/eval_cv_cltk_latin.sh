for p in $(seq 0 8)
do
    datadir=~/cltk_data/user_data/cltk_latin$p
    train=/media/wencan/Private/project/gsoc2016/data/latin_treebank_perseus/latin_training_set$p.pos

    python /media/wencan/Private/project/gsoc2016/cltk_pos/src/pos_cltk_cv.py $train $datadir
done
