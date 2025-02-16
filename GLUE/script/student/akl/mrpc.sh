#!/bin/bash  

TEACHER_PATH=./teacher/exp_distill_mrpc/teacher 
STUDENT_PATH=./student/origin

python main_glue_distill.py --distill_loss kd \
							--do_lower_case \
							--do_train \
							--task_name mrpc \
							--teacher_path $TEACHER_PATH \
							--student_path $STUDENT_PATH \
							--per_gpu_batch_size 16 \
							--num_train_epochs 10 \
							--learning_rate 4e-5 \
							--alpha 0.9 \
							--temperature 4.0 \
							--kl_type akl