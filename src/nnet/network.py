#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:02:56 2018

@author: fenia
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from nnet.init_net import BaseNet
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")


class WalkBasedModel(BaseNet):
    """
    Walk-based Model on Entity graphs.
    """

    def embedding_layer(self, words_, ents_type_, ents_pos_, toks_pos_):
        """
        Embedding layer:
            B: batch size
            E: entities
            D: dimensionality
            W: words
        Associate, words, entity types, positions with vectors.
        Apply dropout to word embeddings
        """
        # words
        w_embed = self.word_embed(words_)  # (B, W, D)

        # positions
        pe_embed = self.pos_embed(ents_pos_)  # (B, E, E, D)
        pt_embed = self.pos_embed(toks_pos_)  # (B, E, W, D)

        # entity types
        te_embed = self.type_embed(ents_type_)  # (B, E, D)
        tt_embed = self.type_embed(torch.tensor([self.o_type]).to(self.device).unsqueeze(0))

        return w_embed, pe_embed, pt_embed, te_embed, tt_embed

    def encoder_layer(self, word_sec, w_embeds):
        """
        BLSTM layer:
            Transform batch of sentences to list
            Pass from BiLSTM
            Pad sequence - form batch again
            Dropout after BLSTM
        """
        ys = self.encoder(torch.split(w_embeds, word_sec.tolist(), dim=0), word_sec)  # (B, W, D)
        return ys

    def merge_tokens(self, info, enc_seq):
        """
        Merge tokens into entities; create binary matrix with indicators for merging
        """
        start, end, w_ids = torch.broadcast_tensors(info[:, :, 2].unsqueeze(-1),
                                                    info[:, :, 3].unsqueeze(-1),
                                                    torch.arange(enc_seq.shape[1])[None, None].to(self.device))

        index_t = (torch.ge(w_ids, start) & torch.lt(w_ids, end)).type('torch.FloatTensor').to(self.device)
        entities = torch.div(torch.matmul(index_t, enc_seq),
                             torch.clamp(torch.sum(index_t, dim=2), 1.0, 100.0).unsqueeze(-1))
        return entities

    def make_pair_indices(self, e_section):
        """
        Construct matrix with a mapping from 3D points -> 1D point
        (batch, row, col) --> pair No
        e.g. [[  0,  1,  2, -1, -1 ],
              [  3,  4,  5, -1, -1 ],
              [  6,  7,  8, -1, -1 ],
              [ -1, -1, -1, -1, -1 ],
              [ -1, -1, -1, -1, -1 ]]
        """
        fshape = (e_section.shape[0], torch.max(e_section).item())

        args_mask = torch.where(torch.lt(torch.arange(fshape[1]).unsqueeze(0).to(self.device), e_section.unsqueeze(1)),
                                torch.ones(fshape).to(self.device), 
                                torch.zeros(fshape).to(self.device))

        cond = torch.matmul(args_mask.unsqueeze(-1), args_mask.unsqueeze(1)).type('torch.ByteTensor').to(self.device)
        bat, rows, cols = torch.nonzero(cond).unbind(dim=1)

        # create a mapping array (to return the flat indice of a certain pair)
        temp = torch.cat([torch.as_tensor([i]).to(self.device).repeat(i) for i in e_section], dim=0)
        # torch.repeat_interleave(e_section, e_section, dim=0).tolist()
        map_pair = torch.split(torch.arange(bat.shape[0]).to(self.device), temp.tolist())
        map_pair = pad_sequence(map_pair, batch_first=True, padding_value=-1)
        map_pair = torch.split(map_pair, e_section.tolist())
        map_pair = pad_sequence(map_pair, batch_first=True, padding_value=-1)
        return bat, rows, cols, cond, map_pair

    def construct_pair(self, info, e_section, enc_out, pe_embed, te_embed):
        """
        Pair representation:
            - Extract entities from BLSTM (average of vectors if > 1 words)
            - pair: (blstm1 + etype1 + pos12, blstm2 + etype2 + pos21)
        """
        # average for words with more than 1 token
        args = self.merge_tokens(info, enc_out)  # (B, E, dim)

        bat, rows, cols, condition, map_pair = self.make_pair_indices(e_section)

        pair_a = torch.cat((args[bat, rows], te_embed[bat, rows], pe_embed[bat, rows, cols]), dim=1)
        pair_b = torch.cat((args[bat, cols], te_embed[bat, cols], pe_embed[bat, cols, rows]), dim=1)

        pairs = torch.cat((pair_a, pair_b), dim=1)  # (BEE, dim)
        return bat, rows, cols, args, pairs, condition, map_pair

    def find_word_context(self, info, word_section, bat, rows, cols):
        """
        Create mask for context words of each pair
        """
        # context tokens (remaining words in the sentence)
        start, end, \
        w_ids, w_sec = torch.broadcast_tensors(info[:, :, 2].unsqueeze(-1),
                                               info[:, :, 3].unsqueeze(-1),
                                               torch.arange(torch.max(word_section).item())[None, None].to(self.device),
                                               word_section.unsqueeze(1).repeat((1, info.shape[1])).unsqueeze(-1))

        toks = (torch.lt(w_ids, start) | torch.ge(w_ids, end))  # target ents
        w_pad = torch.lt(w_ids, w_sec)
        tmp_ = torch.all(toks, dim=1)  # all entities excluded
        cntx_toks = (tmp_[:, None] & w_pad)
        return cntx_toks[bat, rows]

    def find_entity_context(self, info, ent_section, bat, rows, cols):
        """
        Create mask for context entities of each pair
        """
        e_ids, o_ids, \
        e_sec = torch.broadcast_tensors(info[:, :, 0].unsqueeze(-1),
                                        torch.arange(info.shape[1])[None, None].to(self.device),  # total
                                        ent_section.unsqueeze(1).repeat((1, info.shape[1])).unsqueeze(-1))

        e1 = torch.ne(e_ids[bat, rows], o_ids[bat, rows]) & torch.lt(o_ids[bat, rows], e_sec[bat, rows])
        e2 = torch.ne(e_ids[bat, cols], o_ids[bat, cols]) & torch.lt(o_ids[bat, cols], e_sec[bat, cols])
        cntx_ents = (e1 & e2)
        return cntx_ents  # (BEE, E)

    def construct_context(self, bat, rows, cols, info, ent_section, word_section, args, enc_out,
                          pe_embed, pt_embed, te_embed, tt_embed, map_pair):
        """
        Form context for each target pair: word + type_word + pos_word_E1 + pos_word_E2
        'map_pair' is unnecessary, used for debugging
        """
        ct_mask = self.find_word_context(info, word_section, bat, rows, cols)
        ce_mask = self.find_entity_context(info, ent_section, bat, rows, cols)

        # matrix
        context_ents = torch.cat((args[bat], te_embed[bat], pe_embed[bat, rows], pe_embed[bat, cols]), dim=2)
        tt_embed, _ = torch.broadcast_tensors(tt_embed, torch.zeros((bat.shape[0], enc_out.shape[1], 1)))
        context_toks = torch.cat((enc_out[bat], tt_embed, pt_embed[bat, rows], pt_embed[bat, cols]), dim=2)

        context_ents = torch.where(ce_mask.unsqueeze(2), context_ents, torch.zeros_like(context_ents))
        context_toks = torch.where(ct_mask.unsqueeze(2), context_toks, torch.zeros_like(context_toks))
        
        context4pairs = torch.cat((context_ents, context_toks), dim=1)  # (BEE, E+W, dim)
        return context4pairs, torch.cat((ce_mask, ct_mask), dim=1)

    def classification(self, l2r_, gtruth_, pairs, map_pair):
        """
        Softmax classifier
        - separate classification of L2R and R2L pairs
        """
        # Separate pairs into: left-to-right and right-to-left, self relations (e.g. AA) not included
        l2r = torch.nonzero(torch.ne(l2r_, -1)).unbind(dim=1)

        # separate gold labels
        l2r_truth = gtruth_[l2r[0], l2r[1], l2r[2]]
        r2l_truth = gtruth_[l2r[0], l2r[2], l2r[1]]
        
        # predictions
        l2r_pairs = pairs[map_pair[l2r[0], l2r[1], l2r[2]]]
        r2l_pairs = pairs[map_pair[l2r[0], l2r[2], l2r[1]]]

        if self.direction == 'l2r':
            l2r_pairs = self.classifier(l2r_pairs)
            loss = F.cross_entropy(l2r_pairs, l2r_truth)
            probs, preds = F.softmax(l2r_pairs, dim=1).detach_().max(dim=1)
            
        elif self.direction == 'r2l':
            r2l_pairs = self.classifier(r2l_pairs)
            loss = F.cross_entropy(r2l_pairs, r2l_truth)
            reverse = self.reverse_labels()
            probs, b = F.softmax(r2l_pairs, dim=1).detach_().max(dim=1)
            preds = reverse[b]

        elif self.direction == 'l2r+r2l':
            l2r_pairs = self.classifier(l2r_pairs)
            r2l_pairs = self.classifier(r2l_pairs)
            loss = F.cross_entropy(l2r_pairs, l2r_truth) + F.cross_entropy(r2l_pairs, r2l_truth)
            probs, preds = self.correct_predictions(F.softmax(l2r_pairs, dim=1).detach_(),
                                                    F.softmax(r2l_pairs, dim=1).detach_())

        else:
            print('Wrong directionality selection!')
            sys.exit(0)

        return loss, probs, preds, l2r_truth

    def forward(self, binp):
        """
        Forward computation
        1. embedding layer
        2. encoder (BLSTM) layer
        3. Pair representation layer
            + Context representation
            + Attention
            + Linear layer for dimensionality reduction
        4. Walk Generation layer
        5. Classification layer
        """
        # 1. Embedding layer
        w_embed, pe_embed, pt_embed, te_embed, tt_embed = \
            self.embedding_layer(binp['text'], binp['ents'][:, :, 1], binp['pos_ee'], binp['pos_et'])

        # 2. Encoder (BLSTM) layer
        enc_out = self.encoder_layer(binp['word'], w_embed)
     
        # 3. Pair representation layer
        bat, rows, cols, \
        args, pairs, condition, map_pair = self.construct_pair(binp['ents'], binp['entity'], enc_out,
                                                               pe_embed, te_embed)

        if self.att:
            context, mask = self.construct_context(bat, rows, cols, binp['ents'], binp['entity'], binp['word'], args,
                                                   enc_out, pe_embed, pt_embed, te_embed, tt_embed, map_pair)

            # ATTENTION on context of every pair --> arg1 + arg2 + context (BEE, D)
            context, scores = self.attention(context, mask=mask)

            # Target pair representation
            pairs = torch.cat((pairs, context), dim=1)

        # reduce dimensionality of target pairs representations
        pairs = self.reduce(pairs)
        
        # 4. Walks
        if self.walks_iter > 0:
            pairs = self.walk_layer(pairs, condition, map_pair)

        # 5. Classification + Loss
        loss, probs, preds, truth = self.classification(binp['l2r'], binp['rels'], pairs, map_pair)
        stats_ = self.measure_statistics(preds, truth)

        if self.att:
            return loss, stats_, probs, preds, truth, scores
        else:
            return loss, stats_, probs, preds, truth, torch.zeros_like(scores)

    def measure_statistics(self, *inputs):
        """
        Calculate: True Positives (TP), False Positives (FP), False Negatives (FN)
        GPU & CPU code
        """
        y, t = inputs

        label_num = torch.as_tensor([self.sizes['rel_size']]).long().to(self.device)
        ignore_label = torch.as_tensor([self.lab2ign]).long().to(self.device)

        mask_t = torch.eq(t, ignore_label).view(-1)        # true = no_relation
        mask_p = torch.eq(y, ignore_label).view(-1)        # pred = no_relation

        true = torch.where(mask_t, label_num, t.view(-1))  # t: ground truth labels (replace ignored with +1)
        pred = torch.where(mask_p, label_num, y.view(-1))  # y: output of neural network (replace ignored with +1)

        tp_mask = torch.where(torch.eq(pred, true), true, label_num)
        fp_mask = torch.where(torch.ne(pred, true), pred, label_num)  # this includes wrong positive classes as well
        fn_mask = torch.where(torch.ne(pred, true), true, label_num)

        tp = torch.bincount(tp_mask, minlength=self.sizes['rel_size'] + 1)[:self.sizes['rel_size']]
        fp = torch.bincount(fp_mask, minlength=self.sizes['rel_size'] + 1)[:self.sizes['rel_size']]
        fn = torch.bincount(fn_mask, minlength=self.sizes['rel_size'] + 1)[:self.sizes['rel_size']]
        tn = torch.sum(mask_t & mask_p)
        return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    def reverse_labels(self):
        labmap = []
        for e in range(0, self.sizes['rel_size']):
            x_ = self.maps['idx2rel'][e].split(':')
            if x_[1] == 'NR':
                labmap += [self.maps['rel2idx']['1:NR:2']]
            else:
                labmap += [self.maps['rel2idx'][x_[2] + ':' + x_[1] + ':' + x_[0]]]
        return torch.tensor(labmap).long().to(self.device)

    def correct_predictions(self, even_pred, odd_pred):
        """
        Correct predictions: From 2 direction relations choose
            - if reverse labels -> keep one of them
            - if one positive, one negative -> keep the positive
            - if different labels -> more confident (highest probability)
        """
        labmap = self.reverse_labels()
        lab2ign = torch.as_tensor([self.lab2ign]).long().to(self.device)

        # split predictions into 2 arrays: relations (even) & inv-relations (odd)
        even_probs, even_lb = torch.max(even_pred, dim=1)
        odd_probs, odd_lb = torch.max(odd_pred, dim=1)
        inv_odd_lb = labmap[odd_lb]

        minus = torch.full(even_probs.shape, -1).long().to(self.device)

        # if inverse of one-another (e.g. 1:rel:2 & 2:rel:1 of both NR) (this is correct) --> keep them the L2R label
        x1 = torch.where(torch.eq(even_lb, inv_odd_lb), even_lb, minus)
        x1_p = torch.where(torch.eq(even_lb, inv_odd_lb), even_probs, minus.float())

        # if both are positive with different labels --> choose from probability
        cond = torch.ne(even_lb, lab2ign) & torch.ne(odd_lb, lab2ign) & torch.ne(even_lb, inv_odd_lb)
        xa = torch.where(cond, even_probs, minus.float())
        xb = torch.where(cond, odd_probs, minus.float())

        x2 = torch.where(torch.ge(xa, xb) & torch.ne(xa, minus.float()) & torch.ne(xb, minus.float()), even_lb, minus)
        x3 = torch.where(torch.lt(xa, xb) & torch.ne(xa, minus.float()) & torch.ne(xb, minus.float()), inv_odd_lb, minus)
        x2_p = torch.where(torch.ge(xa, xb) & torch.ne(xa, minus.float()) & torch.ne(xb, minus.float()), even_probs, minus.float())  
        x3_p = torch.where(torch.lt(xa, xb) & torch.ne(xa, minus.float()) & torch.ne(xb, minus.float()), odd_probs, minus.float())

        # if one positive & one negative --> choose the positive
        x4 = torch.where(torch.eq(even_lb, lab2ign) & torch.ne(odd_lb, lab2ign), inv_odd_lb, minus)
        x4_p = torch.where(torch.eq(even_lb, lab2ign) & torch.ne(odd_lb, lab2ign), odd_probs, minus.float())
        x5 = torch.where(torch.ne(even_lb, lab2ign) & torch.eq(odd_lb, lab2ign), even_lb, minus)
        x5_p = torch.where(torch.ne(even_lb, lab2ign) & torch.eq(odd_lb, lab2ign), even_probs, minus.float())

        fin = torch.stack([x1, x2, x3, x4, x5], dim=0)
        fin_p = torch.stack([x1_p, x2_p, x3_p, x4_p, x5_p], dim=0)

        assert (torch.sum(torch.clamp(fin, -1.0, 0.0), dim=0) == -4).all(), "EVALUATION: error"
        assert (torch.sum(torch.clamp(fin_p, -1.0, 0.0), dim=0) == -4).all(), "EVALUATION: error"
        fin_preds = torch.max(fin, dim=0)[0]
        fin_probs = torch.max(fin_p, dim=0)[0]
        return fin_probs, fin_preds

