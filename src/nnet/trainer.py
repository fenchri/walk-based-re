#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:02:56 2018

@author: fenia
"""

import torch
import numpy as np
import random
from torch import autograd, nn, optim
from nnet.network import WalkBasedModel
from utils import *
import time
import datetime
import copy
import os
import json
from converter import concat_examples
np.set_printoptions(threshold=np.inf)


class Trainer:
    def __init__(self, data, params, loader, model_folder):
        self.converter = concat_examples
        self.data = data
        self.device = torch.device("cuda:{}".format(params['gpu']) if params['gpu'] != -1 else "cpu")
        self.params = params
        self.loader = loader
        self.epoch = params['epoch']
        self.primary_metric = params['early_metric']
        self.es = params['early_stopping']
        self.pa = params['param_avg']
        self.show_class = params['show_class']
        self.saveto = os.path.join(model_folder, params['save_preds'])
        self.model_folder = model_folder

        self.best_score = 0.0
        self.best_epoch = 0
        self.best_loss = 9999999999
            
        if params['early_stopping']:
            self.patience = params['patience']
            self.cur_patience = 0
                        
        if params['param_avg']:
            self.averaged_params = {}

        self.train_res = {'loss': [], 'score': []}
        self.test_res = {'loss': [], 'score': []}

        self.model = self.init_model()
        self.optimizer = self.set_optimizer()
        print_options(self.model, self.params)
        print_hyperparams(self.model)

    def init_model(self):
        model = WalkBasedModel(self.params,
                               {'word_size': self.loader.n_words, 'pos_size': self.loader.n_pos,
                               'type_size': self.loader.n_type, 'rel_size': self.loader.n_rel},
                               self.loader.pre_embeds,
                               lab2ign=self.loader.label2ignore,
                               o_type=self.loader.type2index['O'],
                               maps={'idx2word': self.loader.index2word, 'word2idx': self.loader.word2index,
                                     'idx2rel': self.loader.index2rel, 'rel2idx': self.loader.rel2index,
                                     'idx2pos': self.loader.index2pos, 'pos2idx': self.loader.pos2index,
                                     'idx2type': self.loader.index2type, 'type2idx': self.loader.type2index})

        # GPU/CPU
        if self.params['gpu'] != -1:
            torch.cuda.set_device(self.device)
            model.to(self.device)
        return model

    def set_optimizer(self):
        # OPTIMIZER
        # do not regularize biases
        params2reg = []
        params0reg = []
        for p_name, p_value in self.model.named_parameters():
            if '.bias' in p_name:
                params0reg += [p_value]
            else:
                params2reg += [p_value]
        assert len(params0reg) + len(params2reg) == len(list(self.model.parameters()))
        groups = [dict(params=params2reg), dict(params=params0reg, weight_decay=.0)]
        optimizer = optim.Adam(groups, lr=self.params['lr'], weight_decay=self.params['reg'], amsgrad=True)

        # Train Model
        print()
        for p_name, p_value in self.model.named_parameters():
            if p_value.requires_grad:
                print(p_name)
        return optimizer

    @staticmethod
    def iterator(x, shuffle_=False, batch_size=1):
        """
        Create a new iterator for this epoch.
        Shuffle the data if specified.
        """
        if shuffle_:
            random.shuffle(x)
        new = [x[i:i+batch_size] for i in range(0, len(x), batch_size)]
        return new

    def early_stopping(self, epoch):
        """
        Perform early stopping.
        If performance does not improve for a number of consecutive epochs ( == "patience")
        then stop the training and keep the best epoch: stopped_epoch - patience

        Args:
            epoch (int): current training epoch

        Returns: (int) best_epoch, (bool) stop
        """
        if self.test_res['score'][-1] > self.best_score:  # improvement of primary metric
            self.best_score = self.test_res['score'][-1]
            self.best_epoch = epoch
            self.cur_patience = 0
            save_model(self.model_folder, self.model, self.loader)
            
        #if self.test_res['loss'][-1] < self.best_loss:
        #    self.best_loss = self.test_res['loss'][-1]
        else:
            self.cur_patience += 1

        if self.patience == self.cur_patience:  # early stop must take place
            self.best_epoch = epoch - self.patience
            return True
        else:
            return False

    def parameter_averaging(self, epoch=None, reset=False):
        """
        Perform parameter averaging.
        For each epoch, average the parameters up to this epoch and then evaluate on test set.
        Args:
            'reset' option: use the last epoch parameters for the next epoch
            'epoch' given: estimate the average until this epoch
        """
        for p_name, p_value in self.model.named_parameters():
            if p_name not in self.averaged_params:
                self.averaged_params[p_name] = []

            if reset:
                p_new = copy.deepcopy(self.averaged_params[p_name][-1])  # use last epoch param

            elif epoch:
                p_new = np.mean(self.averaged_params[p_name][:epoch], axis=0)  # estimate average until this epoch

            else:
                self.averaged_params[p_name].append(p_value.data.to('cpu').numpy())
                p_new = np.mean(self.averaged_params[p_name], axis=0)  # estimate average

            # assign to array
            if self.device != 'cpu':
                p_value.data = torch.from_numpy(p_new).to(self.device)
            else:
                p_value.data = torch.from_numpy(p_new)

    def run(self):
        """
        Run main training process.
        """
        print('\n======== START TRAINING: {} ========\n'.format(
            datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S")))

        random.shuffle(self.data['train'])  # shuffle training data at least once
        
        for epoch in range(1, self.epoch + 1):
            self.train_epoch(epoch)

            if self.pa:
                self.parameter_averaging()  # use parameter averaging on the eval set

            self.eval_epoch()

            stop = self.early_stopping(epoch)  # early stopping criterion
            if self.es and stop:
                break

            if self.pa:
                self.parameter_averaging(reset=True)

        print('Best epoch: {}'.format(self.best_epoch))
        if self.pa:
            self.parameter_averaging(epoch=self.best_epoch)
        self.eval_epoch(final=True, save_predictions=True)

        print('\n======== END TRAINING: {} ========\n'.format(
            datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S")))

    def train_epoch(self, epoch):
        """
        Train model on the training set for 1 epoch, estimate performance and average loss.
        """
        t1 = time.time()
        output_tr = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'loss': [], 'preds': [], 'truth': [], 'probs': []}

        self.model = self.model.train()
        train_iter = self.iterator(self.data['train'], batch_size=self.params['batch'], shuffle_=True)
            
        for batch in train_iter:
            batch = self.convert_batch(batch)

            with autograd.detect_anomaly():
                self.optimizer.zero_grad()

                loss, stats, probs, preds, truth, att_scores = self.model(batch)
                output_tr['preds'] += preds.to('cpu').data.tolist()
                output_tr['probs'] += probs.to('cpu').data.tolist()
                output_tr['truth'] += truth.to('cpu').data.tolist()
                output_tr['loss'] += [loss.item()]
                output_tr['tp'] += [stats['tp'].to('cpu').data.numpy()]
                output_tr['fp'] += [stats['fp'].to('cpu').data.numpy()]
                output_tr['fn'] += [stats['fn'].to('cpu').data.numpy()]
                output_tr['tn'] += [stats['tn'].to('cpu').data.numpy()]

            loss.backward()          # backward computation
            nn.utils.clip_grad_norm_(self.model.parameters(), self.params['gc'])  # gradient clipping
            self.optimizer.step()    # update

        t2 = time.time()

        # estimate performance
        total_loss, scores = self.performance(output_tr)
        self.train_res['loss'] += [total_loss]
        self.train_res['score'] += [scores[self.primary_metric]]

        print('Epoch: {:02d} | TRAIN | LOSS = {:.04f}'.format(epoch, total_loss), end="")
        print_results(scores, self.show_class, t2-t1)

    def eval_epoch(self, final=False, save_predictions=False):
        """
        Evaluate model on the test set for one epoch, estimate performance and average loss.
        Args:
            final: Final model evaluation
            save_predictions: save (or not) ... predictions :)
        """
        t1 = time.time()
        output_ts = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'loss': [], 'preds': [], 'truth': [], 'probs': []}

        save_attention = []
        pids = []
        self.model = self.model.eval()
        test_iter = self.iterator(self.data['test'], batch_size=self.params['batch'], shuffle_=False)
        
        for batch_ in test_iter:

            for b in batch_:
                to_keep = np.where(b['l2r'] != -1)
                pids.extend(b['info'][to_keep])

            batch_ = self.convert_batch(batch_)
            with torch.no_grad():
                loss, stats, probs, preds, truth, att_scores = self.model(batch_)

                output_ts['preds'] += preds.to('cpu').data.tolist()
                output_ts['probs'] += probs.to('cpu').data.tolist()
                output_ts['truth'] += truth.to('cpu').data.tolist()
                output_ts['loss'] += [loss.item()]
                output_ts['tp'] += [stats['tp'].to('cpu').data.numpy()]
                output_ts['fp'] += [stats['fp'].to('cpu').data.numpy()]
                output_ts['fn'] += [stats['fn'].to('cpu').data.numpy()]
                output_ts['tn'] += [stats['tn'].to('cpu').data.numpy()]

        t2 = time.time()

        # estimate performance
        total_loss, scores = self.performance(output_ts)

        if not final:
            self.test_res['loss'] += [total_loss]
            self.test_res['score'] += [scores[self.primary_metric]]
        print('            TEST  | LOSS = {:.04f}'.format(total_loss), end="")
        print_results(scores, self.show_class, t2-t1)
        print()

        if save_predictions:
            write_pred2file(output_ts['preds'], output_ts['probs'], pids, self.saveto, self.loader.index2rel)
            write_errors2file(output_ts['preds'], pids, self.saveto, self.loader.index2rel)
            write_bingo2file(output_ts['preds'], pids, self.saveto, self.loader.index2rel)

    def performance(self, stats):
        """
        Estimate performance: micro and macro average precision, recall, F1 score.
        CPU based
        """
        def lab_map(a):
            tmp = self.loader.index2rel[a]
            tmp = tmp.split(':')
            if tmp[1] == 'NR':
                return self.loader.rel2index['1:NR:2']
            else:
                return self.loader.rel2index[tmp[2]+':'+tmp[1]+':'+tmp[0]]

        def fbeta_score(precision, recall, beta=1.0):
            beta_square = beta * beta
            if (precision != 0.0) and (recall != 0.0):
                res = ((1 + beta_square) * precision * recall / (beta_square * precision + recall))
            else:
                res = 0.0
            return res

        def micro_scores(all_tp, all_fp, all_fn, all_tn):
            atp = np.sum(all_tp)
            afp = np.sum(all_fp)
            afn = np.sum(all_fn)
            atn = np.sum(all_tn)
            micro_p = (1.0 * atp) / (atp + afp) if (atp + afp != 0) else 0.0
            micro_r = (1.0 * atp) / (atp + afn) if (atp + afn != 0) else 0.0
            micro_f = fbeta_score(micro_p, micro_r)
            return micro_p, micro_r, micro_f, atp, afp, afn, atn

        def macro_scores(all_tp, all_fp, all_fn):
            ctp = []
            cfp = []
            cfn = []

            if ':' in list(self.loader.rel2index.keys())[-1]:
                seen = []
                for i in range(0, self.loader.n_rel):
                    if i == self.loader.label2ignore:
                        continue
                    elif (i in seen) or (lab_map(i) in seen):
                        continue
                    else:
                        ctp.append(all_tp[i] + all_tp[lab_map(i)])
                        cfp.append(all_fp[i] + all_fp[lab_map(i)])
                        cfn.append(all_fn[i] + all_fn[lab_map(i)])
                        seen.append(i)
                        seen.append(lab_map(i))
            else:
                for i in range(0, self.loader.n_rel):
                    if i == self.loader.label2ignore:
                        continue
                    else:
                        ctp.append(all_tp[i])
                        cfp.append(all_fp[i])
                        cfn.append(all_fn[i])

            pp = []
            rr = []
            for j in range(0, len(ctp)):
                pp.append((1.0 * ctp[j]) / (ctp[j] + cfp[j]) if (ctp[j] + cfp[j]) != 0 else 0.0)
                rr.append((1.0 * ctp[j]) / (ctp[j] + cfn[j]) if (ctp[j] + cfn[j]) != 0 else 0.0)
            assert len(pp) == len(rr)

            macro_p = np.mean(pp)
            macro_r = np.mean(rr)
            macro_f = fbeta_score(macro_p, macro_r)
            return  macro_p, macro_r, macro_f

        def accuracy(atp, afp, afn, atn):
            return (atp + atn) / (atp + atn + afp + afn) if (atp + atn + afp + afn) else 0.0

        def prf1(all_tp, all_fp, all_fn, all_tn):
            assert len(all_tp) == len(all_fp) == len(all_fn) == len(all_tn)

            all_tp = np.sum(all_tp, axis=0)  # sum per class for all batches
            all_fp = np.sum(all_fp, axis=0)
            all_fn = np.sum(all_fn, axis=0)
            all_tn = np.sum(all_tn, axis=0)

            micro_p, micro_r, micro_f, \
            atp, afp, afn, atn = micro_scores(all_tp, all_fp, all_fn, all_tn)
            macro_p, macro_r, macro_f = macro_scores(all_tp, all_fp, all_fn)
            acc = accuracy(atp, afp, afn, atn)

            return {'acc': acc,
                    'micro_p': micro_p, 'micro_r': micro_r, 'micro_f': micro_f,
                    'macro_p': macro_p, 'macro_r': macro_r, 'macro_f': macro_f,
                    'tp': atp, 'actual': atp + afn, 'pred': atp + afp, 'total': len(stats['preds']),
                    'per_class': []}

        fin_loss = sum(stats['loss']) / len(stats['loss'])
        scores = prf1(stats['tp'], stats['fp'], stats['fn'], stats['tn'])
        return fin_loss, scores

    def convert_batch(self, batch):
        # TODO faster
        batch = [{key: value for key, value in b_.items() if key != 'info' and key != 'sentId'} for b_ in batch]

        converted_batch = concat_examples(batch, device=self.device, padding=-1)

        if self.params['example']:
            for i, b in enumerate(batch):
                print('===== DOCUMENT NO {} ====='.format(i))
                print(' '.join([self.loader.index2word[t] for t in b['text']]))
                print(b['ents'])
                print(b['rels'])
                print(np.array([self.loader.index2pos[t] for t in
                                b['pos_ee'].ravel()]).reshape(-1, b['pos_ee'].shape[0], b['pos_ee'].shape[1]))
                print(np.array([self.loader.index2pos[t] for t in
                                b['pos_et'].ravel()]).reshape(-1, b['pos_et'].shape[0], b['pos_et'].shape[1]))
                print()
            sys.exit()

        converted_batch['text'] = converted_batch['text'][converted_batch['text'] != -1]
        converted_batch['ents'][:, :, 1][converted_batch['ents'][:, :, 1] == -1] = self.loader.n_type  # mask padded
        converted_batch['pos_ee'][converted_batch['pos_ee'] == -1] = self.loader.n_pos
        converted_batch['pos_et'][converted_batch['pos_et'] == -1] = self.loader.n_pos
        return converted_batch
