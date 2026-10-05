__all__ = ['TrackTimerCB', 'TrackTrainingCB', 'PrintResultsCB', 'TerminateOnNaNCB',
            'TrackerCB', 'SaveModelCB', 'EarlyStoppingCB']

from ..basics import *
from .core import Callback
import torch
import time
import numpy as np
from pathlib import Path
from src.callback.patch_mask import *


class TrackTimerCB(Callback):
    def __init__(self):
        super().__init__()


    def before_fit(self):
        self.learner.epoch_time = None

    def before_epoch_train(self):
        self.start_time = time.time()

    def after_epoch_train(self):
        self.learner.epoch_time = self.format_time(time.time() - self.start_time)

    def format_time(self, t):
        "Format `t` (in seconds) to (h):mm:ss"
        t = int(t)
        h, m, s = t // 3600, (t // 60) % 60, t % 60
        if h != 0:
            return f'{h}:{m:02d}:{s:02d}'
        else:
            return f'{m:02d}:{s:02d}'


class TrackTrainingCB(Callback):

    def __init__(self, train_metrics=False, valid_metrics=True, test_metrics=True):
        super().__init__()
        self.train_metrics, self.valid_metrics, self.test_metrics = train_metrics, valid_metrics, test_metrics

    def init_cb_(self):
        self.setup()
        self.initialize_recorder()
        if hasattr(self.loss_func, 'reduction'):
            self.mean_reduction_ = True if self.loss_func.reduction == 'mean' else False

    def before_fit(self):
        self.setup()
        self.initialize_recorder()
        if hasattr(self.loss_func, 'reduction'):
            self.mean_reduction_ = True if self.loss_func.reduction == 'mean' else False

    def setup(self):
        self.valid_loss = True
        self.test_loss = True
        if self.learner.dls:
            if not self.learner.dls.valid: self.valid_metrics = False
            else: self.valid_loss = True
        if self.metrics:
            if not isinstance(self.metrics, list): self.metrics = [self.metrics]
            self.metric_names = [func.__name__ for func in self.metrics]
        else: self.metrics, self.metric_names = [], []

    def initialize_recorder(self):
        recorder = {'epoch': [],  'train_loss': [],  'train_mse': []}
        if self.valid_loss:
            recorder['valid_loss'] = []
            recorder['valid_mse'] = []
        if self.test_loss:
            recorder['test_loss'] = []
            recorder['test_mse'] = []
        for name in self.metric_names:
            if self.train_metrics: recorder['train_'+name] = []
            if self.valid_metrics: recorder['valid_'+name] = []
            if self.test_metrics: recorder['test_'+name] = []
        self.recorder = recorder
        self.learner.recorder = recorder


    def initialize_batch_recorder(self, with_metrics):
        batch_recorder = {'n_samples': [], 'batch_losses': [], 'batch_mses': [],'with_metrics': with_metrics}
        self.batch_recorder = batch_recorder

    def reset(self):
        # Only reset preds and targs if do_test is True
        if getattr(self.learner, 'do_test', False):
            self.targs, self.preds = [],[]
        self.n_samples = 0
        self.batch_loss = []


    def after_epoch(self):
        self.recorder['epoch'].append(self.epoch)
        self.learner.recorder = self.recorder

    def before_epoch_train(self):
        # define storage for batch training loss and metrics        
        self.initialize_batch_recorder(with_metrics=self.train_metrics)
        self.reset()

    def before_epoch_valid(self):
        # if valid data is available, define storage for batch training loss and metrics
        # if self.dls.valid:  self.initialize_batch_recorder(with_metrics=self.valid_metrics)
        self.initialize_batch_recorder(with_metrics=self.valid_metrics)
        self.reset()
    
    def before_epoch_test(self):
        # if valid data is available, define storage for batch training loss and metrics
        # if self.dls.valid:  self.initialize_batch_recorder(with_metrics=self.valid_metrics)
        self.initialize_batch_recorder(with_metrics=self.test_metrics)
        self.reset()


    def after_epoch_train(self):
        values = self.compute_scores()
        # save training loss after one epoch                
        self.recorder['train_loss'].append( values['loss'] )
        self.recorder['train_mse'].append( values['mse'] )
        # save metrics after one epoch         
        if self.train_metrics:
            for name, func in zip(self.metric_names, self.metrics):
                self.recorder['train_'+name].append( values[name] )


    def after_epoch_valid(self):
        # if there is no valid data, don't store
        if not self.learner.dls.valid: return
        values = self.compute_scores()
        # save training loss after one epoch
        self.recorder['valid_loss'].append( values['loss'] )
        self.recorder['valid_mse'].append( values['mse'] )
        # save metrics after one epoch         
        if self.valid_metrics:
            for name, func in zip(self.metric_names, self.metrics):
                self.recorder['valid_'+name].append( values[name] )

    def after_epoch_test(self):
        # if there is no test data, don't store
        if not self.learner.dls.test: return
        values = self.compute_scores()
        # save training loss after one epoch
        self.recorder['test_loss'].append( values['loss'] )
        self.recorder['test_mse'].append( values['mse'] )
        # save metrics after one epoch         
        if self.test_metrics:
            for name, func in zip(self.metric_names, self.metrics):
                self.recorder['test_'+name].append( values[name] )

    def after_batch_train(self): self.accumulate('train')  # save batch recorder
    def after_batch_valid(self): self.accumulate('valid')  # save batch recorder
    def after_batch_test(self):
        self.before_epoch_test() 
        self.before_fit()
        self.accumulate('test')  # save batch recorder

    def accumulate(self, phase):
        if self.loss is None:
            return # skip if there is no loss
        # Handle multiple return values from the model
        if len(self.batch) > 2:
            xb, yb = self.batch[:2]  # Only take the first two values
        else:
            xb, yb = self.batch
        bs = len(xb)

        self.batch_recorder['n_samples'].append(bs)
        
        # get batch loss 
        loss = self.loss.detach()*bs if self.mean_reduction_ else self.loss.detach()
        mse = self.mse.detach()*bs if self.mean_reduction_ else self.loss.detach()
        self.batch_recorder['batch_losses'].append(loss)
        self.batch_recorder['batch_mses'].append(mse)
        # print iter loss
        if self.iter % 500 == 0:
            n = sum(self.batch_recorder['n_samples'])
            cur_loss = sum(self.batch_recorder['batch_losses']).item()/n
            cur_mse = sum(self.batch_recorder['batch_mses']).item()/n  # averaging

        if yb is None: self.batch_recorder['with_metrics'] = False
        if len(self.metrics) == 0: self.batch_recorder['with_metrics'] = False
        # Only accumulate predictions and targets if do_test is True
        if self.batch_recorder['with_metrics'] and getattr(self.learner, 'do_test', False):
            if not isinstance(self.preds, list):
                self.preds = []
            if not isinstance(self.targs, list):
                self.targs = []
            self.preds.append(self.pred.detach().cpu())
            if (self.pred.shape != yb.shape):
                yb, num_patch = create_patch(yb, self.pred.shape[3], self.pred.shape[3])
            self.targs.append(yb.detach().cpu())



    def compute_scores(self):
        "calculate losses and metrics after each epoch"
        values = {}
        # calculate loss after each epoch        
        n = sum(self.batch_recorder['n_samples'])   # get total number of samples
        values['loss'] = sum(self.batch_recorder['batch_losses']).item()/n  # averaging
        values['mse'] = sum(self.batch_recorder['batch_mses']).item()/n  # averaging
        # Only compute scores for preds and targs if do_test is True
        if getattr(self.learner, 'do_test', False) and len(self.preds) > 0:
            if isinstance(self.preds, list):
                self.preds = torch.cat(self.preds)
            if isinstance(self.targs, list):
                self.targs = torch.cat(self.targs)
        return values


class TerminateOnNaNCB(Callback):
    " A callback to stop the training if loss is NaN"
    def after_batch_train(self):
        if torch.isinf(self.loss) or torch.isnan(self.loss): raise KeyboardInterrupt


class PrintResultsCB(Callback):
    def __init__(self):
        super().__init__()

    def get_header(self, recorder):
        "recorder is a dictionary"
        header = ['epoch', 'train_loss', 'train_mse']
        if self.learner.dls.valid:
            header.extend(['valid_loss', 'valid_mse'])
        if self.learner.dls.test and self.learner.do_test:
            header.extend(['test_loss', 'test_mse'])
        return header+['time']

    def before_fit(self):
        if self.run_finder: return          # don't print if lr_finder is called
        if not hasattr(self.learner, 'recorder'): return      # don't print if there is no recorder
        header = self.get_header(self.learner.recorder)
        self.print_header = '{:>15s}'*len(header)
        self.print_value = '{:>15d}' + '{:>15.6f}'*(len(header)-2) + '{:>15}'
        print(self.print_header.format(*header))

    def after_epoch(self):
        if self.run_finder: return      # don't print if lr_finder is called
        if not hasattr(self.learner, 'recorder'): return           # don't print if there is no recorder
        epoch_logs = []
        for key in self.get_header(self.learner.recorder)[:-1]:  # exclude 'time'
            value=self.learner.recorder[key][-1] if self.learner.recorder[key] else None
            epoch_logs += [value]
        if self.learner.epoch_time: epoch_logs.append(self.learner.epoch_time)
        # print('epoch_logs', epoch_logs)
        print(self.print_value.format(*epoch_logs))



class TrackerCB(Callback):
    def __init__(self, monitor='train_loss', comp=None, min_delta=0.):
        super().__init__()
        if comp is None: comp = np.less if 'loss' in monitor or 'error' in monitor or 'mse' in monitor else np.greater
        if comp == np.less: min_delta *= -1
        self.monitor, self.comp, self.min_delta = monitor, comp, min_delta

    def before_fit(self):
        if self.run_finder: return
        if self.best is None: self.best = float('inf') if self.comp == np.less else -float('inf')
        self.monitor_names = list(self.learner.recorder.keys())
        assert self.monitor in self.monitor_names

    def after_epoch(self):
        if self.run_finder: return
        val = self.learner.recorder[self.monitor][-1]
        if self.comp(val - self.min_delta, self.best): self.best, self.new_best = val,True
        else: self.new_best = False


class SaveModelCB(TrackerCB):
    def __init__(self, monitor='train_loss', comp=None, min_delta=0.,
                        every_epoch=False, fname='model', path=None, with_opt=False, save_process_id=0, global_rank=None):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta)
        self.every_epoch = every_epoch
        self.last_saved_path = None
        self.path, self.fname = path, fname
        self.with_opt = with_opt
        self.save_process_id = save_process_id

        # Identify the worker that saves the model to a file: check if the process' global_rank == save_process_id
        # If running locally using either a cpu/gpu without using DDP -> set save_process_id = global_rank
        # Else if running in DDP mode but user doesn't specify global_rank -> global_rank = current_device
        #       (local_rank 0 from each node will save the model)
        # Else if user provides the global_rank -> use the global_rank to check

        if  global_rank:
            self.global_rank = global_rank
        else:
            if torch.cuda.is_available():
                self.global_rank = torch.cuda.current_device()
                if not torch.distributed.is_initialized():
                    self.save_process_id = self.global_rank
            else:
                self.global_rank = 0


    def _save(self, fname, path):
        if self.global_rank == self.save_process_id:
            self.last_saved_path = self.learner.save(fname, path, with_opt=self.with_opt)

    def after_epoch(self):
        if self.every_epoch:
            if ((self.epoch%self.every_epoch) == 0) or (self.epoch==self.n_epochs-1):
                self._save(f'{self.fname}_{self.epoch}', self.path)
        else:
            super().after_epoch()
            if self.new_best:
                print(f'Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.')
                self._save(f'{self.fname}', self.path)

    def after_fit(self):
        if self.run_finder: return
        if not self.every_epoch and self.global_rank == self.save_process_id:
            self.learner.load(self.last_saved_path, with_opt=self.with_opt)


class EarlyStoppingCB(TrackerCB):
    def __init__(self, monitor='train_loss', comp=None, min_delta=0,
                        patient=5):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta)
        self.patient = patient

    def before_fit(self):
        # set the impatient level
        self.impatient_level = 0
        super().before_fit()

    def after_epoch(self):
        super().after_epoch()
        if self.new_best: self.impatient_level = 0   # reset the impatience
        else:
            self.impatient_level += 1
            if self.impatient_level > self.patient:
                print(f'No improvement since epoch {self.epoch-self.impatient_level}: early stopping')
                raise KeyboardInterrupt



