for p in $(seq 0 8)
do
    datadir=~/cltk_data/user_data/latin_lapos$p
    train=/media/wencan/Private/project/gsoc2016/data/latin_treebank_perseus/latin_training_set$p.pos
    
    python /media/wencan/Private/project/gsoc2016/cltk_pos/src/pos_lapos_cv.py $train $datadir 3
done


