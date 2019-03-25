import matplotlib.pyplot as plt
import numpy as np
inp_f = ['5_1_RNN.txt', '5_1_GRU.txt', '5_1_TRANSFORMER.txt']

for i, f_name in enumerate(inp_f):
    f = open(f_name)
    lines = [line[:-1] for line in f.readlines()]
    name, number = lines[0], lines[1:]
    number = [float(n) for n in number]
    f.close()

    plt.plot(np.arange(len(number)), number, label=name)

plt.xlabel('Time step')
plt.ylabel('Average loss')
plt.legend()
plt.title('Average loss over each time step t')
plt.savefig('./fig_5_1.jpg')
