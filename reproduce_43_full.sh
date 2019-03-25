# For Full(40 epochs) experiment

# ---------- RNN -----------
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=50 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best

python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=20 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best 

python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=15 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best 

python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=30 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best 

# ----------- GRU ----------------

python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=750 --num_layers=3 --dp_keep_prob=0.35  --save_best 

python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=1 --dp_keep_prob=0.45  --save_best 

python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=50 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35  --save_best 

python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=40 --seq_len=35 --hidden_size=1500 --num_layers=1 --dp_keep_prob=0.45  --save_best 

python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=40 --seq_len=50 --hidden_size=1500 --num_layers=1 --dp_keep_prob=0.45  --save_best 




# ----------- Transformer ----------------

python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9 --save_best 

python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=50 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9 --save_best 

python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=64 --seq_len=60 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9 --save_best 




