from cuml.manifold import TSNE
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import os
from PIL import Image

colors = ['#F0F8FF','#FAEBD7','#00FFFF','#7FFFD4','#F0FFFF','#F5F5DC','#FFE4C4',
            '#000000','#FFEBCD','#0000FF','#8A2BE2','#A52A2A','#DEB887','#5F9EA0',
            '#7FFF00','#D2691E','#FF7F50','#6495ED','#FFF8DC']

parser = argparse.ArgumentParser(description='Visualize features by t-SNE')
parser.add_argument('--perplexity', type=float, default=30.0)
parser.add_argument('--n_iter', type=int, default=1000)
parser.add_argument('--n_iter_without_progress', type=int, default=300)
parser.add_argument('--angle', type=float, default=0.5)
parser.add_argument('--method', type=str, default='fft')
parser.add_argument('--n_neighbors', type=int, default=90)
parser.add_argument('--mask', type=str, required=True)
args = parser.parse_args()
perplexity = args.perplexity
n_iter = args.n_iter
n_iter_without_progress = args.n_iter_without_progress
angle = args.angle
method = args.method
n_neighbors = args.n_neighbors


TSNE_func = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, n_iter_without_progress=n_iter_without_progress, angle=angle, method=method, n_neighbors=n_neighbors)

x_input = torch.load("figure_tensor.pt") # shape: [h, w, D] D is dimension of figure tensor for each pixel
D = x_input.shape[2]

x_input = torch.reshape(x_input, (-1, D)).squeeze()

#y = torch.randint(0, 19, [x_tsne.shape[0]])
y = Image.open(args.mask)
y = torch.tensor(np.array(y))

y = torch.reshape(y, (1, -1)).squeeze().cpu()

x_in = []
x_in = torch.tensor(x_in).cuda()

color_show = []

for i in range(len(y)):
    if y[i] <= 19:
        x_in = torch.cat((x_in, torch.reshape(x_input[i], (1,-1))))
        color_show.append(y[i].item())
    if i % 10000 == 0:
        print(f"enumerate {i}")
        
print(x_in.shape)
print(D)

color_show = torch.tensor(color_show)
    
x_tsne = TSNE_func.fit_transform(x_in)

picture = plt.figure(figsize=(10,10),dpi=80)

#pic_co = [colors[i] for i in y]

p_x = x_tsne[:,0]
p_y = x_tsne[:,1]

p_x = np.array(p_x.get())
p_y = np.array(p_y.get())

print(f"len of p_x is {len(p_x)}")

'''
show_x = []
show_y = []
show_color = []

for i in range(len(p_x)):
    if y[i] <= 19:
        show_x.append(p_x[i])
        show_y.append(p_y[i])
        show_color.append(colors[y[i]])
'''
plt.scatter(p_x, p_y, color=show_color)

plt.savefig(f'perplexity-{perplexity}-n_iter-{n_iter}-n_iter_without_progress-{n_iter_without_progress}-angle-{angle}-method-{method}-n_neighbors-{n_neighbors}.png')
