#!/bin/bash  

TEACHER_PATH=./teacher/exp_distill_qqp/teacher 
STUDENT_PATH=./student/origin

python main_glue_distill.py --distill_loss kd_anneal+saliency \
							--do_lower_case \
							--do_train \
							--task_name qqp \
							--teacher_path $TEACHER_PATH \
							--student_path $STUDENT_PATH \
							--per_gpu_batch_size 32 \
							--num_train_epochs 8 \
							--learning_rate 4e-5 \
							--alpha 1.0 \
							--temperature 4.0 \ 
							--saliency_w 50 \
							--topk 734