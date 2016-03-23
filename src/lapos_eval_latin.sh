datadir=~/cltk_data/user_data/latin_lapos_1
lapos_home=../../lapos-0.1.2/

python /media/wencan/Private/project/gsoc2016/cltk_pos/src/pos_lapos_cv.py /media/wencan/Private/project/gsoc2016/data/latin_treebank_perseus/latin_training_set_1.pos $datadir 1

for i in $(seq 0 9)
do
    echo $i    
    #train a model for each fold
    
    mkdir $datadir/model$i
    $lapos_home/lapos-learn -m $datadir/model$i $datadir/train_$i.pos
    
    #test the model
    $lapos_home/lapos -m $datadir/model$i < $datadir/test_$i.txt > $datadir/test_$i.result

    #evaluate the model for the test fold
    $lapos_home/lapos-eval $datadir/test_$i.pos $datadir/test_$i.result $datadir/train_$i.pos 2> $datadir/test_$i.out
done
