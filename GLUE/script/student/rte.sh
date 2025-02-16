#!/bin/bash  

TEACHER_PATH=./teacher/exp_distill_rte/teacher 
STUDENT_PATH=./student/origin

python main_glue_distill.py --distill_loss kd+saliency \
							--do_lower_case \
							--do_train \
							--task_name rte \
							--teacher_path $TEACHER_PATH \
							--student_path $STUDENT_PATH \
							--per_gpu_batch_size 16 \
							--num_train_epochs 12 \
							--learning_rate 2e-5 \
							--alpha 0.9 \
							--temperature 2.0 \ 
							--saliency_w 10 \
							--topk 700