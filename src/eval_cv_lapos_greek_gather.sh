language=greek

for p in $(seq 0 8)
do
    datadir=~/cltk_data/user_data/$language\_lapos_translate$p
    train=../../data/$language\_treebank_perseus/$language\_training_set$p.pos.translate
    
    python pos_lapos_cv.py $train $datadir 3
done


