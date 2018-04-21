from torch.autograd import Variable
import numpy as np
import torch
from data_util import config

def get_input_from_batch(batch, use_cuda):
  enc_lens_idx = np.argsort(batch.enc_lens)[::-1]

  batch_size = len(enc_lens_idx)

  enc_batch = Variable(torch.from_numpy(batch.enc_batch[enc_lens_idx]).long())
  enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask[enc_lens_idx])).float()
  enc_lens = batch.enc_lens[enc_lens_idx]
  extra_zeros = None
  enc_batch_extend_vocab = None

  if config.pointer_gen:
    enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab[enc_lens_idx]).long())
    # max_art_oovs is the max over all the article oov list in the batch
    if batch.max_art_oovs > 0:
      extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

  c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

  coverage = None
  if config.is_coverage:
    coverage = Variable(torch.zeros(enc_batch.size()))

  if use_cuda:
    enc_batch = enc_batch.cuda()
    enc_padding_mask = enc_padding_mask.cuda()

    if enc_batch_extend_vocab is not None:
      enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
    if extra_zeros is not None:
      extra_zeros = extra_zeros.cuda()
    c_t_1 = c_t_1.cuda()

    if coverage is not None:
      coverage = coverage.cuda()

  return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage

def get_output_from_batch(batch, use_cuda):
  enc_lens_idx = np.argsort(batch.enc_lens)[::-1]

  dec_batch = Variable(torch.from_numpy(batch.dec_batch[enc_lens_idx]).long())
  dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask[enc_lens_idx])).float()
  dec_lens = batch.dec_lens[enc_lens_idx]
  max_dec_len = np.max(dec_lens)
  dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()

  target_batch = Variable(torch.from_numpy(batch.target_batch[enc_lens_idx])).long()

  if use_cuda:
    dec_batch = dec_batch.cuda()
    dec_padding_mask = dec_padding_mask.cuda()
    dec_lens_var = dec_lens_var.cuda()
    target_batch = target_batch.cuda()


  return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

