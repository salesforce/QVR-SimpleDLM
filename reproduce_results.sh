MODEL_PATH=$1
DATA_PATH=$2
OUTPUT_DIR=$MODEL_PATH

CUDA_VISIBLE_DEVICES=0 python run_query_value_retrieval.py --model_type simpledlm --model_name_or_path $MODEL_PATH \
--data_dir $DATA_PATH --output_dir $OUTPUT_DIR --do_eval