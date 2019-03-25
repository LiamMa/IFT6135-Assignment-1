### !/bin/python
### coding: utf-8

# Code outline/scaffold for
# ASSIGNMENT 2: RNNs, Attention, and Optimization
# By Tegan Maharaj, David Krueger, and Chin-Wei Huang
# IFT6135 at University of Montreal
# Winter 2019
#
# based on code from:
#    https://github.com/deeplearningathome/pytorch-language-model/blob/master/reader.py
#    https://github.com/ceshine/examples/blob/master/word_language_model/main.py
#    https://github.com/teganmaharaj/zoneout/blob/master/zoneout_word_ptb.py
#    https://github.com/harvardnlp/annotated-transformer

# GENERAL INSTRUCTIONS:
#    - ! IMPORTANT!
#      Unless we're otherwise notified we will run exactly this code, importing
#      your models from models.py to test them. If you find it necessary to
#      modify or replace this script (e.g. if you are using TensorFlow), you
#      must justify this decision in your report, and contact the TAs as soon as
#      possible to let them know. You are free to modify/add to this script for
#      your own purposes (e.g. monitoring, plotting, further hyperparameter
#      tuning than what is required), but remember that unless we're otherwise
#      notified we will run this code as it is given to you, NOT with your
#      modifications.
#    - We encourage you to read and understand this code; there are some notes
#      and comments to help you.
#    - Typically, all of your code to submit should be written in models.py;
#      see further instructions at the top of that file / in TODOs.
#          - RNN recurrent unit
#          - GRU recurrent unit
#          - Multi-head attention for the Transformer
#    - Other than this file and models.py, you will probably also write two
#      scripts. Include these and any other code you write in your git repo for
#      submission:
#          - Plotting (learning curves, loss w.r.t. time, gradients w.r.t. hiddens)
#          - Loading and running a saved model (computing gradients w.r.t. hiddens,
#            and for sampling from the model)

# PROBLEM-SPECIFIC INSTRUCTIONS:
#    - For Problems 1-3, paste the code for the RNN, GRU, and Multi-Head attention
#      respectively in your report, in a monospace font.
#    - For Problem 4.1 (model comparison), the hyperparameter settings you should run are as follows:
#          --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
#          --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
#          --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best
#    - In those experiments, you should expect to see approximately the following
#      perplexities:
#                  RNN: train:  120  val: 157
#                  GRU: train:   65  val: 104
#          TRANSFORMER:  train:  67  val: 146
#    - For Problem 4.2 (exploration of optimizers), you will make use of the
#      experiments from 4.1, and should additionally run the following experiments:
#          --model=RNN --optimizer=SGD --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
#          --model=GRU --optimizer=SGD --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
#          --model=TRANSFORMER --optimizer=SGD --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=.9
#          --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=1 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.35
#          --model=GRU --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
#          --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9
#    - For Problem 4.3 (exloration of hyperparameters), do your best to get
#      better validation perplexities than the settings given for 4.1. You may
#      try any combination of the hyperparameters included as arguments in this
#      script's ArgumentParser, but do not implement any additional
#      regularizers/features. You may (and will probably want to) run a lot of
#      different things for just 1-5 epochs when you are trying things out, but
#      you must report at least 3 experiments on each architecture that have run
#      for at least 40 epochs.
#    - For Problem 5, perform all computations / plots based on saved models
#      from Problem 4.1. NOTE this means you don't have to save the models for
#      your exploration, which can make things go faster. (Of course
#      you can still save them if you like; just add the flag --save_best).
#    - For Problem 5.1, you can modify the loss computation in this script
#      (search for "LOSS COMPUTATION" to find the appropriate line. Remember to
#      submit your code.
#    - For Problem 5.3, you must implement the generate method of the RNN and
#      GRU.  Implementing this method is not considered part of problems 1/2
#      respectively, and will be graded as part of Problem 5.3


import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy

np = numpy

# NOTE ==============================================
# This is where your models are imported
from models import RNN, GRU
from models import make_model as TRANSFORMER

##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')

# Arguments you may need to set to run different experiments in 4.1 & 4.2.
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus. We suggest you change the default\
                    here, rather than passing as an argument, to avoid long file paths.')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (RNN, GRU, TRANSFORMER)')
parser.add_argument('--optimizer', type=str, default='SGD_LR_SCHEDULE',
                    help='optimization algo to use; SGD, SGD_LR_SCHEDULE, ADAM')
parser.add_argument('--seq_len', type=int, default=35,
                    help='number of timesteps over which BPTT is performed')
parser.add_argument('--batch_size', type=int, default=20,
                    help='size of one minibatch')
parser.add_argument('--initial_lr', type=float, default=20.0,
                    help='initial learning rate')
parser.add_argument('--hidden_size', type=int, default=200,
                    help='size of hidden layers. IMPORTANT: for the transformer\
                    this must be a multiple of 16.')
parser.add_argument('--save_best', action='store_true',
                    help='save the model for the best validation performance')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of hidden layers in RNN/GRU, or number of transformer blocks in TRANSFORMER')

# Other hyperparameters you may want to tune in your exploration
parser.add_argument('--emb_size', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs to stop after')
parser.add_argument('--dp_keep_prob', type=float, default=0.35,
                    help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                    (dp_keep_prob=1 means no dropout)')

# Arguments that you may want to make use of / implement more code for
parser.add_argument('--debug', action='store_true')
parser.add_argument('--save_dir', type=str, default='save/',
                    help='path to save the experimental config, logs, model \
                    This is automatically generated based on the command line \
                    arguments you pass and only needs to be set if you want a \
                    custom dir name')
parser.add_argument('--evaluate', action='store_true',
                    help="use this flag to run on the test set. Only do this \
                    ONCE for each model setting, and only after you've \
                    completed ALL hyperparameter tuning on the validation set.\
                    Note we are not requiring you to do this.")

# DO NOT CHANGE THIS (setting the random seed makes experiments deterministic,
# which helps for reproducibility)
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

args = parser.parse_args()
argsdict = args.__dict__
argsdict['code_file'] = sys.argv[0]

# Use the model, optimizer, and the flags passed to the script to make the
# name for the experimental dir
print("\n########## Setting Up Experiment ######################")
flags = [flag.lstrip('--').replace('/', '').replace('\\', '') for flag in sys.argv[1:]]
experiment_path = os.path.join(args.save_dir + '_'.join([argsdict['model'],
                                                         argsdict['optimizer']]
                                                        + flags))

# Increment a counter so that previous results with the same args will not
# be overwritten. Comment out the next four lines if you only want to keep
# the most recent results.
i = 0
while os.path.exists(experiment_path + "_" + str(i)):
    i += 1
experiment_path = experiment_path + "_" + str(i)

# Creates an experimental directory and dumps all the args to a text file
os.mkdir(experiment_path)
print("\nPutting log in %s" % experiment_path)
argsdict['save_dir'] = experiment_path
with open(os.path.join(experiment_path, 'exp_config.txt'), 'w') as f:
    for key in sorted(argsdict):
        f.write(key + '    ' + str(argsdict[key]) + '\n')

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")

###############################################################################
#
# LOADING & PROCESSING


#
###############################################################################

# HELPER FUNCTIONS
def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word


# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)


class Batch:
    "Data processing for the transformer. This class adds a mask to the data."

    def __init__(self, x, pad=-1):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."

        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0

        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# LOAD DATA
print('Loading data from ' + args.data)
raw_data = ptb_raw_data(data_path=args.data)
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))

###############################################################################
#
# MODEL SETUP
#
###############################################################################

# NOTE ==============================================
# This is where your model code will be called. You may modify this code
# if required for your implementation, but it should not typically be necessary,
# and you must let the TAs know if you do so.
model = RNN(emb_size=args.emb_size, hidden_size=args.hidden_size,
            seq_len=args.seq_len, batch_size=args.batch_size,
            vocab_size=vocab_size, num_layers=args.num_layers,
            dp_keep_prob=args.dp_keep_prob)

#
#
# # TODO: implement this class
# class MultiHeadedAttention(nn.Module):
#     def __init__(self, n_heads, n_units, dropout=0.1):
#         """
#         n_heads: the number of attention heads
#         n_units: the number of output units
#         dropout: probability of DROPPING units
#         """
#         super(MultiHeadedAttention, self).__init__()
#         # This sets the size of the keys, values, and queries (self.d_k) to all
#         # be equal to the number of output units divided by the number of heads.
#         self.d_k = n_units // n_heads
#         # This requires the number of n_heads to evenly divide n_units.
#         assert n_units % n_heads == 0
#         self.n_units = n_units
#
#         # TODO: create/initialize any necessary parameters or layers
#         # Initialize all weights and biases uniformly in the range [-k, k],
#         # where k is the square root of 1/n_units.
#         # Note: the only Pytorch modules you are allowed to use are nn.Linear
#         # and nn.Dropout
#
#     def forward(self, query, key, value, mask=None):
#         # TODO: implement the masked multi-head attention.
#         # query, key, and value all have size: (batch_size, seq_len, self.n_units)
#         # mask has size: (batch_size, seq_len, seq_len)
#         # As described in the .tex, apply input masking to the softmax
#         # generating the "attention values" (i.e. A_i in the .tex)
#         # Also apply dropout to the attention values.
#
#         return  # size: (batch_size, seq_len, self.n_units)
#
#
# # ----------------------------------------------------------------------------------
# # The encodings of elements of the input sequence
#
# class WordEmbedding(nn.Module):
#     def __init__(self, n_units, vocab):
#         super(WordEmbedding, self).__init__()
#         self.lut = nn.Embedding(vocab, n_units)
#         self.n_units = n_units
#
#     def forward(self, x):
#         # print (x)
#         return self.lut(x) * math.sqrt(self.n_units)
#
#
# class PositionalEncoding(nn.Module):
#     def __init__(self, n_units, dropout, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, n_units)
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, n_units, 2).float() *
#                              -(math.log(10000.0) / n_units))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + Variable(self.pe[:, :x.size(1)],
#                          requires_grad=False)
#         return self.dropout(x)
#
#
# # ----------------------------------------------------------------------------------
# # The TransformerBlock and the full Transformer
#
#
# class TransformerBlock(nn.Module):
#     def __init__(self, size, self_attn, feed_forward, dropout):
#         super(TransformerBlock, self).__init__()
#         self.size = size
#         self.self_attn = self_attn
#         self.feed_forward = feed_forward
#         self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)
#
#     def forward(self, x, mask):
#         x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # apply the self-attention
#         return self.sublayer[1](x, self.feed_forward)  # apply the position-wise MLP
#
#
# class TransformerStack(nn.Module):
#     """
#     This will be called on the TransformerBlock (above) to create a stack.
#     """
#
#     def __init__(self, layer, n_blocks):  # layer will be TransformerBlock (below)
#         super(TransformerStack, self).__init__()
#         self.layers = clones(layer, n_blocks)
#         self.norm = LayerNorm(layer.size)
#
#     def forward(self, x, mask):
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)
#
#
# class FullTransformer(nn.Module):
#     def __init__(self, transformer_stack, embedding, n_units, vocab_size):
#         super(FullTransformer, self).__init__()
#         self.transformer_stack = transformer_stack
#         self.embedding = embedding
#         self.output_layer = nn.Linear(n_units, vocab_size)
#
#     def forward(self, input_sequence, mask):
#         embeddings = self.embedding(input_sequence)
#         return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)
#
#
# def make_model(vocab_size, n_blocks=6,
#                n_units=512, n_heads=16, dropout=0.1):
#     "Helper: Construct a model from hyperparameters."
#     c = copy.deepcopy
#     attn = MultiHeadedAttention(n_heads, n_units)
#     ff = MLP(n_units, dropout)
#     position = PositionalEncoding(n_units, dropout)
#     model = FullTransformer(
#         transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
#         embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
#         n_units=n_units,
#         vocab_size=vocab_size
#     )
#
#     # Initialize parameters with Glorot / fan_avg.
#     for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
#     return model
#
#









#
#
#
#
# # # # --------------------------------------
# import torch
# import torch.nn as nn
#
# import numpy as np
# import torch.nn.functional as F
# import math, copy, time
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
#
#
#
#
# class RNN(nn.Module):  # Implement a stacked vanilla RNN with Tanh nonlinearities.
#     def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
#         """
#         emb_size:     The number of units in the input embeddings
#         hidden_size:  The number of hidden units per layer
#         seq_len:      The length of the input sequences
#         vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
#         num_layers:   The depth of the stack (i.e. the number of hidden layers at
#                       each time-step)
#         dp_keep_prob: The probability of *not* dropping out units in the
#                       non-recurrent connections.
#                       Do not apply dropout on recurrent connections.
#         """
#         super(RNN, self).__init__()
#         self.emb_size=emb_size
#         self.hidden_size=hidden_size
#         self.seq_len=seq_len
#         self.vocab_size=vocab_size
#         self.num_layers=num_layers
#         self.dp_keep_prob=dp_keep_prob
#         self.batch_size=batch_size
#
#         # embedding layer
#         self.embedding=nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=emb_size)
#
#         # fc layer
#         ls=[nn.Linear(emb_size,hidden_size)]
#         ls+=[nn.Linear(hidden_size,hidden_size) for _ in range(num_layers-1)]
#         self.W_x=nn.ModuleList(ls)
#
#         self.w_y=nn.Linear(hidden_size,vocab_size) # vocab + <eos> + <unk>
#
#         self.W_h=nn.ModuleList([nn.Linear(hidden_size,hidden_size) for _ in range(num_layers)])
#
#         self.init_weights()
#         self.tanh=nn.Tanh()
#
#
#     def init_weights(self):
#         self.embedding.weight.data.uniform_(-0.1, 0.1)
#
#         self.w_y.weight.data.uniform_(-0.1,0.1)
#         self.w_y.bias.data.fill_(0)
#
#         for i in self.W_x:
#             k=i.weight.data.size(0)
#             k=(1/k)**(1/2)
#             i.weight.data.uniform_(-k,k)
#             i.bias.data.uniform_(-k,k)
#
#         for i in self.W_h:
#             k=i.weight.data.size(0)
#             k=(1/k)**(1/2)
#             i.weight.data.uniform_(-k,k)
#             i.bias.data.uniform_(-k,k)
#
#     def init_hidden(self):
#         # TODO ========================
#         # initialize the hidden states to zero
#
#         hidden = Variable(torch.Tensor(np.zeros((self.num_layers, self.batch_size, self.hidden_size))),
#                           requires_grad=True)
#         """
#         This is used for the first mini-batch in an epoch, only.
#         """
#         """
#         Q: How to deal with a smaller last mini-batch?
#         """
#
#         # return  # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
#         return hidden
#
#     def forward(self, inputs, hidden):
#         # if input sequences length different, do padding
#         logits=torch.Tensor(np.zeros([1,self.batch_size,self.vocab_size])).requires_grad_(True)
#         for index in range(inputs.size(0)):
#             input=inputs[index]
#             input=input.long()
#             input=self.embedding(input)
#             for hl in range(self.num_layers):
#                 w_x=self.W_x[hl]
#                 w_h=self.W_h[hl]
#                 input=self.tanh(w_x(input)+w_h(hidden[hl]))
#                 hidden[hl]=input
#             output=self.w_y(hidden[-1])
#             logits=torch.cat((logits,output.unsqueeze(0)),dim=0)
#
#         logits=logits[1:]
#         return logits.view(self.seq_len, self.batch_size, self.vocab_size),hidden
#
#
#
# # Problem 2
# class GRU(nn.Module):  # Implement a stacked GRU RNN
#     """
#     Follow the same instructions as for RNN (above), but use the equations for
#     GRU, not Vanilla RNN.
#     """
#
#     def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
#         super(GRU, self).__init__()
#         self.emb_size = emb_size
#         self.hidden_size = hidden_size
#         self.seq_len = seq_len
#         self.vocab_size = vocab_size
#         self.num_layers = num_layers
#         self.dp_keep_prob = dp_keep_prob
#         self.batch_size = batch_size
#
#         # embedding layer
#         self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=emb_size)
#
#         # fc layer
#         ls_r = [copy.deepcopy(nn.Linear(emb_size, hidden_size))]
#         ls_r += [copy.deepcopy(nn.Linear(hidden_size, hidden_size)) for _ in range(num_layers - 1)]
#         ls_z = [copy.deepcopy(nn.Linear(emb_size, hidden_size))]
#         ls_z += [copy.deepcopy(nn.Linear(hidden_size, hidden_size)) for _ in range(num_layers - 1)]
#         ls_h = [copy.deepcopy(nn.Linear(emb_size, hidden_size))]
#         ls_h += [copy.deepcopy(nn.Linear(hidden_size, hidden_size)) for _ in range(num_layers - 1)]
#         # list store reference, not value
#
#         self.W_r = nn.ModuleList(ls_r)
#         self.W_z = nn.ModuleList(ls_z)
#         self.W_h = nn.ModuleList(ls_h)
#         self.U_h = nn.ModuleList([copy.deepcopy(nn.Linear(hidden_size, hidden_size)) for _ in range(num_layers)])
#         self.U_r = nn.ModuleList([copy.deepcopy(nn.Linear(hidden_size, hidden_size)) for _ in range(num_layers)])
#         self.U_z = nn.ModuleList([copy.deepcopy(nn.Linear(hidden_size, hidden_size)) for _ in range(num_layers)])
#
#         self.w_y = nn.Sequential(
#             nn.Linear(hidden_size, vocab_size),  # vocab # no output of <unk> and <eos>
#             nn.Dropout(1 - self.dp_keep_prob)
#         )
#
#         self.init_weights_uniform()
#         self.sigmoid_r = [copy.deepcopy(nn.Sigmoid()) for _ in range(num_layers)]
#         self.sigmoid_z = [copy.deepcopy(nn.Sigmoid()) for _ in range(num_layers)]
#         self.sigmoid_h = [copy.deepcopy(nn.Sigmoid()) for _ in range(num_layers)]
#         self.tanh = nn.Tanh()
#         # TODO ========================
#
#     def init_weights_uniform(self):
#         self.embedding.weight.data.uniform_(-0.1, 0.1)
#
#         for i in self.w_y:
#             if type(i)== nn.Linear:
#                 i.weight.data.uniform_(-0.1, 0.1)
#                 i.bias.data.fill_(0)
#
#         for i in self.W_r:
#             k = i.weight.data.size(0)
#             k = (1 / k) ** (1 / 2)
#             i.weight.data.uniform_(-k, k)
#             i.bias.data.uniform_(-k, k)
#
#         for i in self.W_z:
#             k = i.weight.data.size(0)
#             k = (1 / k) ** (1 / 2)
#             i.weight.data.uniform_(-k, k)
#             i.bias.data.uniform_(-k, k)
#
#         for i in self.W_h:
#             k = i.weight.data.size(0)
#             k = (1 / k) ** (1 / 2)
#             i.weight.data.uniform_(-k, k)
#             i.bias.data.uniform_(-k, k)
#
#         for i in self.U_r:
#             k = i.weight.data.size(0)
#             k = (1 / k) ** (1 / 2)
#             i.weight.data.uniform_(-k, k)
#             i.bias.data.uniform_(-k, k)
#
#         for i in self.U_z:
#             k = i.weight.data.size(0)
#             k = (1 / k) ** (1 / 2)
#             i.weight.data.uniform_(-k, k)
#             i.bias.data.uniform_(-k, k)
#
#         for i in self.U_h:
#             k = i.weight.data.size(0)
#             k = (1 / k) ** (1 / 2)
#             i.weight.data.uniform_(-k, k)
#             i.bias.data.uniform_(-k, k)
#     # TODO ========================
#
#     def init_hidden(self):
#         hidden = Variable(torch.Tensor(np.zeros((self.num_layers, self.batch_size, self.hidden_size))),
#                           requires_grad=True)
#         return hidden  # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
#
#     def forward(self, inputs, hidden):
#         logits = torch.Tensor(np.zeros([1, self.batch_size, self.vocab_size])).requires_grad_(True)
#         for index in range(inputs.size(0)):
#             input = inputs[index]
#             input = self.embedding(input.long())
#             for hl in range(self.num_layers):
#                 w_r = self.W_r[hl]
#                 w_z = self.W_z[hl]
#                 w_h = self.W_h[hl]
#                 s_r = self.sigmoid_r[hl]
#                 s_z = self.sigmoid_z[hl]
#                 s_h = self.sigmoid_h[hl]
#                 u_h = self.U_h[hl]
#                 u_z = self.U_z[hl]
#                 u_r = self.U_r[hl]
#
#                 r = s_r(w_r(input) + u_r(hidden[hl]))
#                 z = s_z(w_z(input) + u_z(hidden[hl]))
#                 h = s_h(w_h(input) + u_h(torch.mul(r, hidden[hl])))
#                 input = torch.mul((1 - z), hidden[hl]) + torch.mul(z, h)
#                 hidden[hl] = input
#             output = self.w_y(input)
#             logits = torch.cat((logits, output.unsqueeze(0)), dim=0)
#
#         logits = logits[1:]
#
#         # TODO ========================
#         return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden
#
#
# # print("----- build model ------")
#
# #
# #
# # model=GRU(10,20,8,2,10,2,0.5)
# # input=torch.Tensor(np.random.choice(10,[8,2])).requires_grad_(True)
# # hidden=model.init_hidden()
# # logit,hd=model.forward(input,hidden)
# # print(logit.size())
# # print(hd.size())
# #
# # print(model.embedding.weight.data)
# #
# # print(model.w_y.weight.data)
# # print(model.w_y.bias.data)
# #
# # for i in model.W_x:
# #     print(i.weight.data)
# #     print(i.bias.data)
# #
# # for i in model.W_h:
# #     print(i.weight.data)
# #     print(i.bias.data)