language=greek

p=$1
 
datadir=~/cltk_data/user_data/cltk_$language$p
train=/media/wencan/Private/project/gsoc2016/data/$language\_treebank_perseus/$language\_training_set$p.pos

python /media/wencan/Private/project/gsoc2016/cltk_pos/src/pos_cltk_cv.py $train $datadir

