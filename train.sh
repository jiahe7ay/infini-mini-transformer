MODELPATH="./gemma"
DATAPATH="./train_datasets"
MODEL_SIZE="2B"
RUNNAME="infi"
OUTPUTPATH="./"
TOTALBSZ=80
BSZPERDEV=8
DEVICES="0"
NUMGPUS=$(echo $DEVICES | awk -F',' '{print NF}')
GRADACC=$(($TOTALBSZ/$NUMGPUS/$BSZPERDEV))
EPOCHNUM=1
echo "Training  model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BSZPERDEV batch size per GPU, $GRADACC gradient accumulation steps"
#deepspeed --include localhost:$DEVICES --master_port 29502
python ./train.py \
    --model_name_or_path ${MODELPATH} \
    --data_path ${DATAPATH} \
    --output_dir ${OUTPUTPATH}/${RUNNAME} \
    --num_train_epochs ${EPOCHNUM} \
    --per_device_train_batch_size ${BSZPERDEV} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADACC} \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_eval False \
    --evaluation_strategy "no" \
    --lazy_preprocess True \
    --run_name ${RUNNAME} \
    --bf16 True \
   # --deepspeed ./dp_configs/deepspeed_config_zero2_no_offload.json
