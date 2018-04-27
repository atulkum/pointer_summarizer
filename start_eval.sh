export PYTHONPATH=`pwd`
MODEL_PATH=$1
MODEL_NAME=$(basename $MODEL_PATH)
python training_ptr_gen/eval.py $MODEL_PATH >& ../log/eval_log.$MODEL_NAME &

