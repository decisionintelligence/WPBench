# coding=gbk
import os
import random
import torch
import numpy as np
from datetime import datetime

def mae_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def masked_mae_loss(mask_value):
    def loss(preds, labels):
        mae = mae_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae

    return loss





def disp(x, name):
    print(f'{name} shape: {x.shape}')


def get_model_params(model_list):
    model_parameters = []
    for m in model_list:
        if m != None:
            model_parameters += list(m.parameters())
    return model_parameters


def get_log_dir(args):
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    log_dir = os.path.join(current_dir, 'experiments', args.dataset, current_time)
    return log_dir


def load_graph(adj_file, device='cpu'):
    '''Loading graph in form of edge index.'''
    graph = np.load(adj_file)['adj_mx']
    graph = torch.tensor(graph, device=device, dtype=torch.float)
    return graph


def dwa(L_old, L_new, T=2):
    '''
    L_old: list.
    '''
    L_old = torch.tensor(L_old, dtype=torch.float32)
    L_new = torch.tensor(L_new, dtype=torch.float32)
    N = len(L_new)  # task number
    r = L_old / L_new

    w = N * torch.softmax(r / T, dim=0)
    return w.numpy()


def get_project_path():
    project_path = os.path.join(
        os.path.dirname(__file__),
        "..",
    )
    # 将 'STEVE' 改为 'steve'
    project_path = project_path[:find_last(project_path, 'steve') + 5]
    return project_path


def find_last(search, target, start=0):
    loc = search.find(target, start)
    end_loc = loc
    while loc != -1:
        end_loc = loc
        start = loc + 1
        loc = search.find(target, start)
    return end_loc
