#!/bin/bash  

TEACHER_PATH=./teacher/exp_distill_sst-2/teacher 
STUDENT_PATH=./student/origin

python main_glue_distill.py --distill_loss kd \
							--do_lower_case \
							--do_train \
							--task_name sst-2 \
							--teacher_path $TEACHER_PATH \
							--student_path $STUDENT_PATH \
							--per_gpu_batch_size 32 \
							--num_train_epochs 6 \
							--learning_rate 5e-5 \
							--alpha 0.8 \
							--temperature 2.0 \
							--kl_type akl