__all__ = ['Callback', 'SetupLearnerCB', 'GetPredictionsCB', 'GetTestCB', 'SaveResultMemmapCB']


""" 
Callback lists:
    > before_fit
        - before_epoch
            + before_epoch_train                
                ~ before_batch_train
                ~ after_batch_train                
            + after_epoch_train

            + before_epoch_valid                
                ~ before_batch_valid
                ~ after_batch_valid                
            + after_epoch_valid
        - after_epoch
    > after_fit

    - before_predict        
        ~ before_batch_predict
        ~ after_batch_predict          
    - after_predict

"""

from ..basics import *
import torch
import os
import numpy as np
import pandas as pd

DTYPE = torch.float32

class Callback(GetAttr): 
    _default = 'learner'


class SetupLearnerCB(Callback): 
    def __init__(self,device_ids):
        self.device_ids = device_ids       
        self.device = default_device(use_cuda=True, device_ids=device_ids)

    def before_batch_train(self): self._to_device()
    def before_batch_valid(self): self._to_device()
    def before_batch_predict(self): self._to_device()
    def before_batch_test(self): self._to_device()

    def _to_device(self):
        batch = to_device(self.batch, self.device)        
        if self.n_inp > 1: 
            # Handle the case where batch contains more than 2 values
            if len(batch) > 2:
                xb, yb, *rest = batch
                self.learner.batch = (xb, yb, *rest)
            else:
                xb, yb = batch
                self.learner.batch = (xb, yb)
        else: 
            xb, yb = batch, None        
            self.learner.batch = (xb, yb)
        
    def before_fit(self): 
        "Set model to cuda before training"                
        self.learner.model.to(self.device)
        self.learner.device = self.device


class GetPredictionsCB(Callback):
    def __init__(self):
        super().__init__()

    def before_predict(self):
        self.preds = []        
    
    def after_batch_predict(self):        
        # append the prediction after each forward batch           
        self.preds.append(self.pred)

    def after_predict(self):           
        self.preds = torch.concat(self.preds)#.detach().cpu().numpy()

         

class GetTestCB(Callback):
    def __init__(self):
        super().__init__()

    def before_test(self):
        self.preds, self.targets = [], []        
    
    def after_batch_test(self):        
        # append the prediction after each forward batch           
        self.preds.append(self.pred)
        self.targets.append(self.yb)

    def after_test(self):           
        self.preds = torch.concat(self.preds)#.detach().cpu().numpy()
        self.targets = torch.concat(self.targets)#.detach().cpu().numpy()


class SaveResultMemmapCB(Callback):
    def __init__(self, save_path, dataset, original_scale=True, prefix=""):
        super().__init__()
        self.save_path = save_path
        self.dataset = dataset
        self.original_scale = original_scale
        self.prefix = prefix
        self.preds_memmap = None
        self.targets_memmap = None
        self.timestamps_memmap = None
        self.idx = 0

    def before_test(self):
        # Ensure directory exists
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
        n_samples = len(self.dataset)
        pred_len = self.dataset.pred_len
        c = self.dataset.data_y.shape[-1]
        
        # Define file paths
        preds_file = os.path.join(self.save_path, f"{self.prefix}preds.dat")
        targets_file = os.path.join(self.save_path, f"{self.prefix}targets.dat")
        timestamps_file = os.path.join(self.save_path, f"{self.prefix}timestamps.dat")
        
        # Initialize memmaps
        self.preds_memmap = np.memmap(preds_file, dtype='float32', mode='w+', shape=(n_samples, pred_len, c))
        self.targets_memmap = np.memmap(targets_file, dtype='float32', mode='w+', shape=(n_samples, pred_len, c))
        # S26 is enough for 'YYYY-MM-DD HH:MM:SS'
        self.timestamps_memmap = np.memmap(timestamps_file, dtype='S26', mode='w+', shape=(n_samples, pred_len))
        
        self.idx = 0

    def after_batch_test(self):
        preds = self.pred.detach().cpu().numpy()
        targets = self.yb.detach().cpu().numpy()
        batch_size = preds.shape[0]
        
        if self.original_scale:
            shape = preds.shape
            preds = self.dataset.inverse_transform(preds.reshape(-1, shape[-1])).reshape(shape)
            targets = self.dataset.inverse_transform(targets.reshape(-1, shape[-1])).reshape(shape)
            
        self.preds_memmap[self.idx : self.idx + batch_size] = preds
        self.targets_memmap[self.idx : self.idx + batch_size] = targets
        
        # Handle timestamps
        start_indices = np.arange(self.idx, self.idx + batch_size) + self.dataset.seq_len
        pred_len = self.dataset.pred_len
        
        timestamps_batch = []
        for i in range(batch_size):
             s = start_indices[i]
             e = s + pred_len
             # Use .values to get numpy array of datetime64
             ts = self.dataset.date_col_name.iloc[s:e]
             # Format
             if hasattr(ts, 'dt'):
                 ts_str = ts.dt.strftime('%Y-%m-%d %H:%M:%S').values
             else:
                 # If it's already datetime index
                 ts_str = ts.strftime('%Y-%m-%d %H:%M:%S').values
             timestamps_batch.append(ts_str)
             
        self.timestamps_memmap[self.idx : self.idx + batch_size] = np.array(timestamps_batch).astype('S26')
        
        self.idx += batch_size

    def after_test(self):
        if self.preds_memmap is not None: self.preds_memmap.flush()
        if self.targets_memmap is not None: self.targets_memmap.flush()
        if self.timestamps_memmap is not None: self.timestamps_memmap.flush()
        print(f"Saved inference results to {self.save_path}")
