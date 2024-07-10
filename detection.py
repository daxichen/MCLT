import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time

from vit_pytorch import VisionTransformer as vits
from utils import plot_roc_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import laplacian_kernel, sigmoid_kernel
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import pairwise_distances


import moco.builder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sandiego2 189bands AVIRIS
data_name = 'Sandiego2'
# data_name = 'Sandiego100'
# data_name = 'MUUFL'
# data_name = 'Beach4'

# Sandiego2
hyperdata_path = './data/' + data_name + '/sandiego.mat'
gdt_path = './data/' + data_name + '/groundtruth.mat'
prior_path = './data/' + data_name + '/prior_target.mat'
hyperdata = sio.loadmat(hyperdata_path)['data']
hyperdata = hyperdata.reshape(120*120,-1)
hyperdata = hyperdata/1.0
gdt = sio.loadmat(gdt_path)['gt']
gdt = gdt.reshape(-1)
prior = sio.loadmat(prior_path)['prior_target'].T
prior = prior/1.0

# Sandiego100
# hyperdata_path = './data/' + data_name + '/sandiego.mat'
# gdt_path = './data/' + data_name + '/groundtruth.mat'
# prior_path = './data/' + data_name + '/prior_target.mat'
# hyperdata = sio.loadmat(hyperdata_path)['data']
# hyperdata = hyperdata.reshape(100*100,-1)
# hyperdata = hyperdata/1.0
# gdt = sio.loadmat(gdt_path)['groundtruth']
# gdt = gdt.reshape(-1)
# prior = sio.loadmat(prior_path)['prior_target'].T
# prior = prior/1.0

# MUUFL
# hyperdata_path = './data/' + data_name + '/data.mat'
# gdt_path = './data/' + data_name + '/groundtruth.mat'
# prior_path = './data/' + data_name + '/prior_target.mat'
# hyperdata = sio.loadmat(hyperdata_path)['data']
# hyperdata = hyperdata.reshape(325*220,-1)
# hyperdata = hyperdata/1.0
# gdt = sio.loadmat(gdt_path)['groundtruth']
# gdt = gdt.reshape(-1)
# prior = sio.loadmat(prior_path)['prior_target'].T
# prior = prior/1.0

# Beach4
# hyperdata_path = './data/' + data_name + '/data.mat'
# gdt_path = './data/' + data_name + '/groundtruth.mat'
# prior_path = './data/' + data_name + '/prior_target.mat'
# hyperdata = sio.loadmat(hyperdata_path)['data']
# hyperdata = hyperdata.reshape(100*120,-1)
# hyperdata = hyperdata/1.0
# gdt = sio.loadmat(gdt_path)['groundtruth']
# gdt = gdt.reshape(-1)
# prior = sio.loadmat(prior_path)['prior_target'].T
# prior = prior/1.0

def cos_sim(vector_1, vector_2):
    return (9*(10**49))**(np.inner(vector_1, vector_2) / (np.linalg.norm(vector_1) * (np.linalg.norm(vector_2))))

#Sandiego2 model
v = vits(
        img_size=189,
        patch_size=9,
        embed_dim=128,
        depth=2,
        num_heads=8,
        representation_size=None,
        num_classes=128
    )
model = moco.builder.MoCo_ViT(v, dim=128, mlp_dim=256, T=0.07, K=14400, m=0.999)


# Sandiego100
# v = vits(
#         img_size=189,
#         patch_size=9,
#         embed_dim=128,
#         depth=2,
#         num_heads=8,
#         representation_size=None,
#         num_classes=128
#     )
# model = moco.builder.MoCo_ViT(v, dim=128, mlp_dim=256, T=0.07, K=10000, m=0.999)


# MUUFL
# v = vits(
#         img_size=64,
#         patch_size=4,
#         embed_dim=128,
#         depth=2,
#         num_heads=8,
#         representation_size=None,
#         num_classes=128
#     )
# model = moco.builder.MoCo_ViT(v, dim=128, mlp_dim=256, T=0.07, K=13000, m=0.999)

# Beach4
# v = vits(
#         img_size=102,
#         patch_size=6,
#         embed_dim=128,
#         depth=2,
#         num_heads=8,
#         representation_size=None,
#         num_classes=128
#     )
# model = moco.builder.MoCo_ViT(v, dim=128, mlp_dim=256, T=0.07, K=12000, m=0.999)

#Sandiego2
ckpt = torch.load('./save/MCLT/Sandiego2_models/Sandiego2_lr_0.05_bsz_480_temp_0.07_trial_0/ckpt_epoch_50.pth', map_location='cpu')

state_dict = ckpt['model']
if torch.cuda.is_available():
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
model = model.cuda()
model.load_state_dict(state_dict)

start = time.time()
model.eval()
target_detector = []
feature = []
prior = torch.FloatTensor(prior)
dataset = torch.FloatTensor(hyperdata)
with torch.no_grad():
    prior = prior.to(device)
    dataset = dataset.to(device)
    prior_output = model.base_encoder.forward_features(prior)
    dataset_out = model.base_encoder.forward_features(dataset)
    prior_output = prior_output.cuda().data.cpu().numpy()
    dataset_out = dataset_out.cuda().data.cpu().numpy()
    target_detector = (9*(10**49))**cosine_similarity(prior_output, dataset_out)

target_detector = np.array(target_detector).T
max3 = np.amax(target_detector)
min3 = np.amin(target_detector)
target_detector = (target_detector - min3)/(max3 - min3)

target_detector = (target_detector)**60  # Sandiego100=60; Sandiego2=20; Beach4=20; MUUFL=60

end = time.time()
print('running time:', end - start)

plot_roc_curve(gdt, target_detector, data_name)

# Sandiego2
target_detector = np.reshape(target_detector, (120,120))

target_detector = target_detector.tolist()

plt.figure(2)
plt.imshow(target_detector, cmap='afmhot')
plt.axis('off')
pathfigure = './result/' + data_name + '.jpg'
plt.savefig(pathfigure, bbox_inches='tight', pad_inches=0, dpi=600)
plt.show()

path_target_detector = './result/' + data_name + '.mat'
sio.savemat(path_target_detector, {'detect': target_detector})