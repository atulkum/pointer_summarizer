from __future__ import unicode_literals, print_function, division

import os
import time
import sys

import numpy as np
import tensorflow as tf
import torch

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from torch.autograd import Variable

from data_util.utils import calc_running_avg_loss
from model import Model


#TODO
use_cuda = torch.cuda.is_available()

class Evaluate(object):
    def __init__(self, model_file_path):
        self.batch_size = config.batch_size
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, 'eval', self.batch_size, single_pass=True)
        time.sleep(5)
        eval_dir = os.path.join(config.log_root, 'eval')
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        self.summary_writer = tf.summary.FileWriter(eval_dir)

        self.model = Model(model_file_path)

    def eval(self, batch):
        enc_lens_idx = np.argsort(batch.enc_lens)[::-1]

        enc_batch = Variable(torch.from_numpy(batch.enc_batch[enc_lens_idx]).long())
        enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask[enc_lens_idx])).float()
        enc_lens = batch.enc_lens[enc_lens_idx]
        enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab[enc_lens_idx]).long())
        # max_art_oovs is the max over all the article oov list in the batch
        extra_zeros = None
        if batch.max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros((self.batch_size, batch.max_art_oovs)))

        dec_batch = Variable(torch.from_numpy(batch.dec_batch[enc_lens_idx]).long())
        dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask[enc_lens_idx])).float()
        dec_lens = batch.dec_lens[enc_lens_idx]
        dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()
        target_batch = Variable(torch.from_numpy(batch.target_batch[enc_lens_idx])).long()

        c_t_1 = Variable(torch.zeros((self.batch_size, 2 * config.hidden_dim)))

        if use_cuda:
            enc_batch = enc_batch.cuda()
            enc_padding_mask = enc_padding_mask.cuda()
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
            if extra_zeros is not None:
                extra_zeros = extra_zeros.cuda()

            dec_batch = dec_batch.cuda()
            dec_padding_mask = dec_padding_mask.cuda()
            dec_lens_var = dec_lens_var.cuda()
            target_batch = target_batch.cuda()
            c_t_1 = c_t_1.cuda()

        encoder_outputs, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(config.max_dec_steps):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1 = self.model.decoder(y_t_1, s_t_1,
                                                          encoder_outputs, enc_padding_mask, c_t_1,
                                                          extra_zeros, enc_batch_extend_vocab)
            if np.sum(dec_lens > di) == 0:
                break

            gold_logprobs = torch.gather(final_dist, 1, target_batch[:, di].unsqueeze(1)).squeeze()
            step_loss = -gold_logprobs
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_step_losses = sum(step_losses)
        loss = torch.mean(sum_step_losses / dec_lens_var)

        return loss.data[0]

    def run_eval(self):
        start = time.time()
        running_avg_loss, step = 0
        batch = self.batcher.next_batch()
        while batch is not None:
            loss = self.eval(batch)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, step)
            step += 1

            print_interval = 1000
            if step % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (
                    step, print_interval, time.time() - start, loss))
                start = time.time()

            if step % 100 == 0:
                self.summary_writer.flush()

            batch = self.batcher.next_batch()


if __name__ == '__main__':
    model_filename = sys.argv[1]
    eval_processor = Evaluate(model_filename)
    eval_processor.run_eval()


