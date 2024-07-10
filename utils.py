import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.io as sio
import torch.optim as optim
import os

def plot_roc_curve(y, pred, title):
    # ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    fpr, tpr, threshold = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    print('roc_auc:', roc_auc)
    plt.style.use('ggplot')

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    # plt.savefig('result/'+title+'.svg', dpi=600)
    # plt.savefig('result/'+title+ 'ROC' +'.png', bbox_inches= 'tight', pad_inches = 0, dpi = 600)
    plt.savefig('result/' + title + 'ROC' + '.png', bbox_inches='tight', pad_inches=0, dpi=600)
    plt.show()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_optimizer(args, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # optimizer = optim.Adam(model.parameters(),
    #                       lr=opt.learning_rate,
    #                       weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, args, epoch, save_file):
    print('==> Saving...')
    state = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# def loadData(name):
#     data_path = os.path.join(os.getcwd(), 'data')
#     if name == 'Sandiego2':
#         data = sio.loadmat(os.path.join(data_path, 'Sandiego2', 'sandiego.mat'))['data']
#         labels = sio.loadmat(os.path.join(data_path, 'Sandiego2','y1.mat'))['y1']
#         agumentation_one = sio.loadmat(os.path.join(data_path, 'Sandiego2', 'VAE_Features1.mat'))['VAE_Features1']
#         agumentation_two = sio.loadmat(os.path.join(data_path, 'Sandiego2', 'VAE_Features2.mat'))['VAE_Features2']
#     elif name == 'Sandiego100':
#         data = sio.loadmat(os.path.join(data_path, 'Sandiego100', 'sandiego.mat'))['data']
#         labels = sio.loadmat(os.path.join(data_path, 'Sandiego100', 'y1.mat'))['y1']
#         agumentation_one = sio.loadmat(os.path.join(data_path, 'Sandiego100', 'AAE_Features1.mat'))['AAE_Features1']
#         agumentation_two = sio.loadmat(os.path.join(data_path, 'Sandiego100', 'AAE_Features2.mat'))['AAE_Features2']
#     elif name == 'Sandiego':
#         data = sio.loadmat(os.path.join(data_path, 'Sandiego', 'sandiego.mat'))['data']
#         labels = sio.loadmat(os.path.join(data_path, 'Sandiego', 'y1.mat'))['y1']
#         agumentation_one = sio.loadmat(os.path.join(data_path, 'Sandiego', 'AAE_Features1.mat'))['AAE_Features1']
#         agumentation_two = sio.loadmat(os.path.join(data_path, 'Sandiego', 'AAE_Features2.mat'))['AAE_Features2']
#     elif name == 'Synthetic':
#         data = sio.loadmat(os.path.join(data_path, 'Synthetic', 'synthetic.mat'))['data']
#         labels = sio.loadmat(os.path.join(data_path, 'Synthetic', 'y1.mat'))['y1']
#         agumentation_one = sio.loadmat(os.path.join(data_path, 'Synthetic', 'AAE_Features1.mat'))['AAE_Features1']
#         agumentation_two = sio.loadmat(os.path.join(data_path, 'Synthetic', 'AAE_Features2.mat'))['AAE_Features2']
#     elif name == 'HYDICE':
#         data = sio.loadmat(os.path.join(data_path, 'HYDICE','hydice_urban.mat'))['data']
#         labels = sio.loadmat(os.path.join(data_path, 'HYDICE','y1.mat'))['y1']
#         agumentation_one = sio.loadmat(os.path.join(data_path, 'HYDICE', 'AAE_Features1.mat'))['AAE_Features1']
#         agumentation_two = sio.loadmat(os.path.join(data_path, 'HYDICE', 'AAE_Features2.mat'))['AAE_Features2']
#     elif name == 'Airport':
#         data = sio.loadmat(os.path.join(data_path, 'Airport','airport.mat'))['data']
#         labels = sio.loadmat(os.path.join(data_path, 'Airport','y1.mat'))['y1']
#         agumentation_one = sio.loadmat(os.path.join(data_path, 'Airport', 'AAE_Features1.mat'))['AAE_Features1']
#         agumentation_two = sio.loadmat(os.path.join(data_path, 'Airport', 'AAE_Features2.mat'))['AAE_Features2']
#     elif name == 'Airport2':
#         data = sio.loadmat(os.path.join(data_path, 'Airport2','data.mat'))['data']
#         labels = sio.loadmat(os.path.join(data_path, 'Airport2','y1.mat'))['y1']
#         agumentation_one = sio.loadmat(os.path.join(data_path, 'Airport2', 'AAE_Features1.mat'))['AAE_Features1']
#         agumentation_two = sio.loadmat(os.path.join(data_path, 'Airport2', 'AAE_Features2.mat'))['AAE_Features2']
#     elif name == 'Beach':
#         data = sio.loadmat(os.path.join(data_path, 'Beach','data.mat'))['data']
#         labels = sio.loadmat(os.path.join(data_path, 'Beach','y1.mat'))['y1']
#         agumentation_one = sio.loadmat(os.path.join(data_path, 'Beach', 'AAE_Features1.mat'))['AAE_Features1']
#         agumentation_two = sio.loadmat(os.path.join(data_path, 'Beach', 'AAE_Features2.mat'))['AAE_Features2']
#     elif name == 'Beach2':
#         data = sio.loadmat(os.path.join(data_path, 'Beach2','Beach.mat'))['data']
#         labels = sio.loadmat(os.path.join(data_path, 'Beach2','y1.mat'))['y1']
#         agumentation_one = sio.loadmat(os.path.join(data_path, 'Beach2', 'AAE_Features1.mat'))['AAE_Features1']
#         agumentation_two = sio.loadmat(os.path.join(data_path, 'Beach2', 'AAE_Features2.mat'))['AAE_Features2']
#     elif name == 'Urban1':
#         data = sio.loadmat(os.path.join(data_path, 'Urban1','data.mat'))['data']
#         labels = sio.loadmat(os.path.join(data_path, 'Urban1','y1.mat'))['y1']
#         agumentation_one = sio.loadmat(os.path.join(data_path, 'Urban1', 'AAE_Features1.mat'))['AAE_Features1']
#         agumentation_two = sio.loadmat(os.path.join(data_path, 'Urban1', 'AAE_Features2.mat'))['AAE_Features2']
#     elif name == 'Urban2':
#         data = sio.loadmat(os.path.join(data_path, 'Urban2','data.mat'))['data']
#         labels = sio.loadmat(os.path.join(data_path, 'Urban2','y1.mat'))['y1']
#         agumentation_one = sio.loadmat(os.path.join(data_path, 'Urban2', 'AAE_Features1.mat'))['AAE_Features1']
#         agumentation_two = sio.loadmat(os.path.join(data_path, 'Urban2', 'AAE_Features2.mat'))['AAE_Features2']
#     elif name == 'Salinas':
#         data = sio.loadmat(os.path.join(data_path, 'Salinas', 'Salinas_corrected.mat'))['salinas_corrected']
#         labels = sio.loadmat(os.path.join(data_path, 'Salinas', 'gt.mat'))['gt']
#         agumentation_one = sio.loadmat(os.path.join(data_path, 'Salinas', 'agumentation_one.mat'))['agumentation_one']
#         agumentation_two = sio.loadmat(os.path.join(data_path, 'Salinas', 'agumentation_two.mat'))['agumentation_two']
#     elif name == 'Segundo':
#         data = sio.loadmat(os.path.join(data_path, 'Segundo', 'segundo.mat'))['data']
#         labels = sio.loadmat(os.path.join(data_path, 'Segundo', 'y1.mat'))['y1']
#         agumentation_one = sio.loadmat(os.path.join(data_path, 'Segundo', 'AAE_Features1.mat'))['AAE_Features1']
#         agumentation_two = sio.loadmat(os.path.join(data_path, 'Segundo', 'AAE_Features2.mat'))['AAE_Features2']
#     elif name == 'Avon':
#         data = sio.loadmat(os.path.join(data_path, 'Avon', 'avon.mat'))['data']
#         labels = sio.loadmat(os.path.join(data_path, 'Avon', 'y1.mat'))['y1']
#         agumentation_one = sio.loadmat(os.path.join(data_path, 'Avon', 'AAE_Features1.mat'))['AAE_Features1']
#         agumentation_two = sio.loadmat(os.path.join(data_path, 'Avon', 'AAE_Features2.mat'))['AAE_Features2']
#     elif name == 'Cuprite':
#         data = sio.loadmat(os.path.join(data_path, 'Cuprite', 'cuprite.mat'))['data']
#         labels = sio.loadmat(os.path.join(data_path, 'Cuprite', 'y1.mat'))['y1']
#         agumentation_one = sio.loadmat(os.path.join(data_path, 'Cuprite', 'AAE_Features1.mat'))['AAE_Features1']
#         agumentation_two = sio.loadmat(os.path.join(data_path, 'Cuprite', 'AAE_Features2.mat'))['AAE_Features2']
#     return data, labels, agumentation_one, agumentation_two


def loadData(name):
    data_path = os.path.join(os.getcwd(), 'data')
    if name == 'Sandiego2':
        data = sio.loadmat(os.path.join(data_path, 'Sandiego2', 'sandiego.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Sandiego2','groundtruth.mat'))['gt']
    elif name == 'Sandiego100':
        data = sio.loadmat(os.path.join(data_path, 'Sandiego100', 'sandiego.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Sandiego100', 'groundtruth.mat'))['groundtruth']
    elif name == 'MUUFL':
        data = sio.loadmat(os.path.join(data_path, 'MUUFL', 'data.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'MUUFL', 'groundtruth.mat'))['groundtruth']
    elif name == 'Sandiego':
        data = sio.loadmat(os.path.join(data_path, 'Sandiego', 'sandiego.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Sandiego', 'y1.mat'))['y1']
        agumentation_one = sio.loadmat(os.path.join(data_path, 'Sandiego', 'AAE_Features1.mat'))['AAE_Features1']
        agumentation_two = sio.loadmat(os.path.join(data_path, 'Sandiego', 'AAE_Features2.mat'))['AAE_Features2']
    elif name == 'Synthetic':
        data = sio.loadmat(os.path.join(data_path, 'Synthetic', 'synthetic.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Synthetic', 'y1.mat'))['y1']
        agumentation_one = sio.loadmat(os.path.join(data_path, 'Synthetic', 'AAE_Features1.mat'))['AAE_Features1']
        agumentation_two = sio.loadmat(os.path.join(data_path, 'Synthetic', 'AAE_Features2.mat'))['AAE_Features2']
    elif name == 'HYDICE':
        data = sio.loadmat(os.path.join(data_path, 'HYDICE','hydice_urban.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'HYDICE','groundtruth.mat'))['groundtruth']
    elif name == 'Airport':
        data = sio.loadmat(os.path.join(data_path, 'Airport','airport.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Airport','groundtruth.mat'))['groundtruth']
    elif name == 'Airport2':
        data = sio.loadmat(os.path.join(data_path, 'Airport2','data.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Airport2','groundtruth.mat'))['groundtruth']
    elif name == 'Beach':
        data = sio.loadmat(os.path.join(data_path, 'Beach','data.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Beach','y1.mat'))['y1']
        agumentation_one = sio.loadmat(os.path.join(data_path, 'Beach', 'AAE_Features1.mat'))['AAE_Features1']
        agumentation_two = sio.loadmat(os.path.join(data_path, 'Beach', 'AAE_Features2.mat'))['AAE_Features2']
    elif name == 'Beach2':
        data = sio.loadmat(os.path.join(data_path, 'Beach2','Beach.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Beach2','y1.mat'))['y1']
        agumentation_one = sio.loadmat(os.path.join(data_path, 'Beach2', 'AAE_Features1.mat'))['AAE_Features1']
        agumentation_two = sio.loadmat(os.path.join(data_path, 'Beach2', 'AAE_Features2.mat'))['AAE_Features2']
    elif name == 'Urban1':
        data = sio.loadmat(os.path.join(data_path, 'Urban1','data.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Urban1','groundtruth.mat'))['groundtruth']
    elif name == 'Urban2':
        data = sio.loadmat(os.path.join(data_path, 'Urban2','data.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Urban2','y1.mat'))['y1']
        agumentation_one = sio.loadmat(os.path.join(data_path, 'Urban2', 'AAE_Features1.mat'))['AAE_Features1']
        agumentation_two = sio.loadmat(os.path.join(data_path, 'Urban2', 'AAE_Features2.mat'))['AAE_Features2']
    elif name == 'Salinas':
        data = sio.loadmat(os.path.join(data_path, 'Salinas', 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas', 'gt.mat'))['gt']
        agumentation_one = sio.loadmat(os.path.join(data_path, 'Salinas', 'agumentation_one.mat'))['agumentation_one']
        agumentation_two = sio.loadmat(os.path.join(data_path, 'Salinas', 'agumentation_two.mat'))['agumentation_two']
    elif name == 'Segundo':
        data = sio.loadmat(os.path.join(data_path, 'Segundo', 'segundo.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Segundo', 'groundtruth.mat'))['groundtruth']
    elif name == 'Avon':
        data = sio.loadmat(os.path.join(data_path, 'Avon', 'avon.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Avon', 'y1.mat'))['y1']
        agumentation_one = sio.loadmat(os.path.join(data_path, 'Avon', 'AAE_Features1.mat'))['AAE_Features1']
        agumentation_two = sio.loadmat(os.path.join(data_path, 'Avon', 'AAE_Features2.mat'))['AAE_Features2']
    elif name == 'Cuprite':
        data = sio.loadmat(os.path.join(data_path, 'Cuprite', 'cuprite.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Cuprite', 'y1.mat'))['y1']
        agumentation_one = sio.loadmat(os.path.join(data_path, 'Cuprite', 'AAE_Features1.mat'))['AAE_Features1']
        agumentation_two = sio.loadmat(os.path.join(data_path, 'Cuprite', 'AAE_Features2.mat'))['AAE_Features2']
    elif name == 'Urban4':
        data = sio.loadmat(os.path.join(data_path, 'Urban4', 'data.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Urban4', 'groundtruth.mat'))['groundtruth']
    elif name == 'Beach3':
        data = sio.loadmat(os.path.join(data_path, 'Beach3', 'data.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Beach3', 'groundtruth.mat'))['groundtruth']
    elif name == 'Beach4':
        data = sio.loadmat(os.path.join(data_path, 'Beach4', 'data.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Beach4', 'groundtruth.mat'))['groundtruth']
    elif name == 'panelHIMp1':
        data = sio.loadmat(os.path.join(data_path, 'panelHIMp1', 'panelHIM.mat'))['HIM']
        labels = sio.loadmat(os.path.join(data_path, 'panelHIMp1', 'groundtruth.mat'))['groundtruth']

    return data, labels


