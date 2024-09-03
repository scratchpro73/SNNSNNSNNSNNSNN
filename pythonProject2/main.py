import torch
from spikingjelly.activation_based import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt

# x = torch.rand(size=[2, 3])
if_layer = neuron.LIFNode()

if_layer.reset()
x = torch.rand([32]) / 0.9
zer = torch.zeros([32])
T = 150
s_list = []
v_list = []
for t in range(T):
    if t <= 150: s_list.append(if_layer(x).unsqueeze(0))
    else: s_list.append(if_layer(zer).unsqueeze(0))
    v_list.append(if_layer.v.unsqueeze(0))

s_list = torch.cat(s_list)
v_list = torch.cat(v_list)

figsize = (12, 8)
dpi = 200
visualizing.plot_2d_heatmap(array=v_list.numpy(), title='membrane potentials', xlabel='simulating step',
                            ylabel='neuron index', int_x_ticks=True, x_max=T, figsize=figsize, dpi=dpi)


visualizing.plot_1d_spikes(spikes=s_list.numpy(), title='membrane sotentials', xlabel='simulating step',
                        ylabel='neuron index', figsize=figsize, dpi=dpi)

plt.show()