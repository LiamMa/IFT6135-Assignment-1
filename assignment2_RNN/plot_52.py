import matplotlib.pyplot as plt
import numpy as np
inp_f = ['5_2_RNN.txt', '5_2_GRU.txt']

for i, f_name in enumerate(inp_f):
    f = open(f_name)
    lines = [line[:-1] for line in f.readlines()]
    name, number = lines[0], lines[1:]
    number = [float(n) for n in number]
    f.close()

    plt.plot(np.arange(len(number)), number, label=name)

plt.xlabel('Time step')
plt.ylabel('Scaled norm of d L_T / d h_t')
plt.legend()
plt.title('Grad norm scaled by: (y-min(y)) / ((max(y)-min(y))')
plt.savefig('./fig_5_2.jpg')
