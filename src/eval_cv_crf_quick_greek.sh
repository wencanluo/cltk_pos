language=greek

for p in $(seq 0 8)
do
    datadir=~/cltk_data/user_data/$language\_crf_quick$p
    train=/media/wencan/Private/project/gsoc2016/data/$language\_treebank_perseus/$language\_training_set$p.pos

    python /media/wencan/Private/project/gsoc2016/cltk_pos/src/pos_crf_cv_quick.py $train $datadir 1

    for i in $(seq 0 9)
    do
        #echo $i        
        python /media/wencan/Private/project/gsoc2016/cltk_pos/src/pos_crf_cv_quick.py $train $datadir 2 $i
    done
    
done

#python /media/wencan/Private/project/gsoc2016/cltk_pos/src/pos_crf_cv_quick.py $train $datadir 3
