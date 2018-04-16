from __future__ import unicode_literals, print_function, division

import os
import time

import numpy as np
import tensorflow as tf
import torch
from model import Model
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

from custom_adagrad import AdagradCustom

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss

use_cuda = torch.cuda.is_available()

class Train(object):
    def __init__(self):
        self.batch_size = config.batch_size
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=self.batch_size, single_pass=False)
        time.sleep(15)

        train_dir = os.path.join(config.log_root, 'train_%d'%(int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())

        self.optimizer = AdagradCustom(params, lr=config.lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.optimizer.load_state_dict(state['optimizer'])

            start_iter = state['iter']
            start_loss = state['current_loss']

        return start_iter, start_loss

    def train_one_batch(self, batch):
        enc_lens_idx = np.argsort(batch.enc_lens)[::-1]

        enc_batch = Variable(torch.from_numpy(batch.enc_batch[enc_lens_idx]).long())
        enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask[enc_lens_idx])).float()
        enc_lens = batch.enc_lens[enc_lens_idx]
        extra_zeros = None
        enc_batch_extend_vocab = None

        if config.pointer_gen:
            enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab[enc_lens_idx]).long())
            #max_art_oovs is the max over all the article oov list in the batch
            if batch.max_art_oovs > 0:
                extra_zeros = Variable(torch.zeros((self.batch_size, batch.max_art_oovs)))

        dec_batch = Variable(torch.from_numpy(batch.dec_batch[enc_lens_idx]).long())
        dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask[enc_lens_idx])).float()
        dec_lens = batch.dec_lens[enc_lens_idx]
        max_dec_len = np.max(dec_lens)
        dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()

        target_batch = Variable(torch.from_numpy(batch.target_batch[enc_lens_idx])).long()

        c_t_1 = Variable(torch.zeros((self.batch_size, 2 * config.hidden_dim)))

        if use_cuda:
            enc_batch = enc_batch.cuda()
            enc_padding_mask = enc_padding_mask.cuda()

            if enc_batch_extend_vocab is not None:
                enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
            if extra_zeros is not None:
                extra_zeros = extra_zeros.cuda()

            dec_batch = dec_batch.cuda()
            dec_padding_mask = dec_padding_mask.cuda()
            dec_lens_var = dec_lens_var.cuda()
            target_batch = target_batch.cuda()
            c_t_1 = c_t_1.cuda()

        self.optimizer.zero_grad()

        encoder_outputs, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1,  c_t_1 = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss*step_mask
            step_losses.append(step_loss)

        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_step_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)
        loss.backward()

        clip_grad_norm(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.data[0]

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % 100 == 0:
                self.summary_writer.flush()
            print_interval = 1000
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval, time.time() - start, loss))
                start = time.time()
            if iter % 10000 == 0:
                self.save_model(running_avg_loss, iter)

if __name__ == '__main__':
    train_processor = Train()
    train_processor.trainIters(config.max_iterations)
