from __future__ import unicode_literals, print_function, division

#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import os
import time

import torch
from torch.autograd import Variable

from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util import data, config
from model import Model
from data_util.utils import make_html_safe, rouge_eval, rouge_log


use_cuda = torch.cuda.is_available()

class Beam(object):
  def __init__(self, tokens, log_probs, state, attn_dists, p_gens):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.attn_dists = attn_dists
    self.p_gens = p_gens

  def extend(self, token, log_prob, state, attn_dist, p_gen):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      attn_dists = self.attn_dists + [attn_dist],
                      p_gens = self.p_gens + [p_gen])

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path):
        self.batch_size = config.beam_size

        self._decode_dir = os.path.join(config.log_root, 'decode')
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode', batch_size=self.batch_size, single_pass=True)
        time.sleep(5)

        self.model = Model(model_file_path)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


    def decode(self):
        start = time.time()
        counter = 0
        batch = self.batcher.next_batch()
        while batch is not None:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstract_sents = batch.original_abstracts_sents[0]

            self.write_for_rouge(original_abstract_sents, decoded_words, counter)
            counter += 1
            if counter % 10000:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()

            batch = self.batcher.next_batch()

        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)


    def beam_search(self, batch):
        #batch should have only one example
        enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
        if use_cuda:
            enc_batch= enc_batch.cuda()
        enc_lens = batch.enc_lens
        enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
        if use_cuda:
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        max_art_oovs = batch.max_art_oovs

        encoder_hidden = self.model.encoder.initHidden(self.batch_size)
        encoder_outputs, encoder_hidden = self.model.encoder(enc_batch, enc_lens, encoder_hidden)
        encoder_hidden_reduced = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = encoder_hidden_reduced # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()
        #decoder batch preparation, it has beam_size example
        #initially everything repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      attn_dists=[],
                      p_gens=[]) for _ in xrange(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) for t in latest_tokens]
            decoder_input = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                decoder_input = decoder_input.cuda()
            all_state_h =[]
            all_state_c = []
            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

            decoder_hidden = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))

            final_dist, decoder_hidden, attn_dist, p_gen, _ = self.model.decoder(decoder_input,
                                                                             decoder_hidden,
                                                                             encoder_outputs,
                                                                             max_art_oovs,
                                                                             enc_batch_extend_vocab,
                                                                             coverage=None)

            topk_log_probs, topk_ids = torch.topk(final_dist, config.beam_size * 2)

            dec_h, dec_c = decoder_hidden
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in xrange(num_orig_beams):
                h = beams[i]
                for j in xrange(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].data[0],
                                   log_prob=topk_log_probs[i, j].data[0],
                                   state=(dec_h[i], dec_c[i]),
                                   attn_dist=attn_dist[i],
                                   p_gen=p_gen[i])
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

    def write_for_rouge(self, reference_sents, decoded_words, ex_index):
        decoded_sents = []
        while len(decoded_words) > 0:
            try:
                fst_period_idx = decoded_words.index(".")
            except ValueError:
                fst_period_idx = len(decoded_words)
            sent = decoded_words[:fst_period_idx + 1]
            decoded_words = decoded_words[fst_period_idx + 1:]
            decoded_sents.append(' '.join(sent))

        # pyrouge calls a perl script that puts the data into HTML files.
        # Therefore we need to make our output HTML safe.
        decoded_sents = [make_html_safe(w) for w in decoded_sents]
        reference_sents = [make_html_safe(w) for w in reference_sents]

        ref_file = os.path.join(self._rouge_ref_dir, "%06d_reference.txt" % ex_index)
        decoded_file = os.path.join(self._rouge_dec_dir, "%06d_decoded.txt" % ex_index)

        with open(ref_file, "w") as f:
            for idx, sent in enumerate(reference_sents):
                f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
        with open(decoded_file, "w") as f:
            for idx, sent in enumerate(decoded_sents):
                f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")

        print("Wrote example %i to file" % ex_index)

if __name__ == '__main__':
    model_filename = sys.argv[1]
    beam_Search_processor = BeamSearch(model_filename)
    beam_Search_processor.decode()


