#!/bin/bash
#PBS -N GRU_liam
#PBS -A rvj-323-aa
#PBS -l nodes=1:gpus=1
#PBS -o $HOME/RNN/result/job_RNN_2.out
#PBS -e $HOME/RNN/result/job_RNN_2.err
#PBS -l walltime=10:00:00
#PBS -M liheng.ma@mail.mcgill.ca
#PBS -l feature=k80

## load python and install torch
## do not specify the number of processes in helios


module load python/3.6
module load scipy-stack
source $HOME/Liam/bin/activate
cd $HOME/RNN
python -i ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35




