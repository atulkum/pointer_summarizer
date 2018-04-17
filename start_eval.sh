export PYTHONPATH=`pwd`
MODEL=$1
python training_ptr_gen/eval.py $MODEL >& ../log/eval_log &

