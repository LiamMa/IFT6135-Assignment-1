import torch 
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

class RNNCell(nn.Module):

	def __init__(self, input_size, hidden_size, dp_keep_prob):
    
		super(RNNCell, self).__init__()

		self.hidden_size = hidden_size

		self.fc_x = nn.Linear(input_size, hidden_size)
		self.fc_h = nn.Linear(hidden_size, hidden_size)

		self.dropout = nn.Dropout(1 - dp_keep_prob)
		self.tanh = nn.Tanh()


	def init_weights(self):

		k = np.sqrt(1 / self.hidden_size)
		
		nn.init.uniform_(self.fc_x.weight, -k, k)
		nn.init.uniform_(self.fc_x.bias, -k, k)

		nn.init.uniform_(self.fc_h.weight, -k, k)
		nn.init.uniform_(self.fc_h.bias, -k, k)

	def forward(self, inputs, hidden):
    
		inputs_dropout = self.dropout(inputs)

		out = self.fc_x(inputs_dropout) + self.fc_h(hidden)
		out = self.tanh(out)

		return out	

class RNN(nn.Module):

	def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
		
		super(RNN, self).__init__()

		self.emb_size = emb_size
		self.hidden_size = hidden_size
		self.seq_len = seq_len
		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.num_layers = num_layers
		self.dp_keep_prob = dp_keep_prob

		self.dropout = nn.Dropout(1 - dp_keep_prob)

		self.embedding_layer = nn.Embedding(vocab_size, emb_size)

		self.hidden_layers = nn.ModuleList()

		for i in range(num_layers):

			if i == 0:
				self.hidden_layers.append( RNNCell(emb_size, hidden_size, dp_keep_prob) )
			else:
				self.hidden_layers.append( RNNCell(hidden_size, hidden_size, dp_keep_prob) )

		self.output_layer = nn.Linear(hidden_size, vocab_size)

		self.init_weigths()


	def init_weights(self):

		nn.init.uniform_(self.embedding_layer.weight, -0.1, 0.1)

		for hidden_layer in self.hidden_layers:
			hidden_layer.init_weights()

		nn.init.uniform_(self.output_layer.weight, -0.1, 0.1)
		nn.init.constant_(self.output_layer.bias, 0)


	def forward(self, inputs, hidden):

		embedded_inputs = self.embedding_layer(inputs)

		logits = torch.zeros([self.seq_len, self.batch_size, self.vocab_size])

		for t in range(self.seq_len):
			
			hidden_layers_outputs = []

			inputs_l = embedded_inputs[t]

			for l, hidden_layer_l in enumerate(self.hidden_layers):

				hidden_layer_l_output = hidden_layer_l(inputs_l, hidden[l])

				hidden_layers_outputs.append(hidden_layer_l_output)

				inputs_l = hidden_layer_l_output

			hidden = torch.stack(hidden_layers_outputs)

			last_hidden_layer_output_dropout = self.dropout(inputs_l)

			logits[t] = self.output_layer(last_hidden_layer_output_dropout)

		return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden


	def generate(self, input, hidden, generated_seq_len):

		samples = input.view(1, -1)

		embedded_input = self.embedding_layer(samples)
	
		for _ in range(self.seq_len):
			
			hidden_layers_outputs = []

			input_l = embedded_input[0]

			for l, hidden_layer_l in enumerate(self.hidden_layers):

				hidden_layer_l_output = hidden_layer_l(input_l, hidden[l])

				hidden_layers_outputs.append(hidden_layer_l_output)

				input_l = hidden_layer_l_output

			hidden = torch.stack(hidden_layers_outputs)

			last_hidden_layer_output_dropout = self.dropout(input_l)

			logits = self.output_layer(last_hidden_layer_output_dropout)

			token = torch.argmax( nn.Softmax(logits), dim=1 ).detach().view(1, -1)

			samples = torch.cat( (samples, token), dim=0 )

			embedded_input = self.embedding_layer(token)

		return samples