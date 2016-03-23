for p in $(seq 3 8)
do
    datadir=~/cltk_data/user_data/latin_crf_quick_$p
    train=/media/wencan/Private/project/gsoc2016/data/latin_treebank_perseus/latin_training_set_$p.pos

    python /media/wencan/Private/project/gsoc2016/cltk_pos/src/pos_crf_cv_quick.py $train $datadir 1

    for i in $(seq 0 9)
    do
        #echo $i        
        python /media/wencan/Private/project/gsoc2016/cltk_pos/src/pos_crf_cv_quick.py $train $datadir 2 $i
    done
    
done



