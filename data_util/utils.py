#Content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/
import os
import pyrouge
import logging
import tensorflow as tf

def print_results(article, abstract, decoded_output):
  print ("")
  print('ARTICLE:  %s', article)
  print('REFERENCE SUMMARY: %s', abstract)
  print('GENERATED SUMMARY: %s', decoded_output)
  print( "")


def make_html_safe(s):
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s


def rouge_eval(ref_dir, dec_dir):
  r = pyrouge.Rouge155()
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
  rouge_results = r.convert_and_evaluate()
  return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
  log_str = ""
  for x in ["1","2","l"]:
    log_str += "\nROUGE-%s:\n" % x
    for y in ["f_score", "recall", "precision"]:
      key = "rouge_%s_%s" % (x,y)
      key_cb = key + "_cb"
      key_ce = key + "_ce"
      val = results_dict[key]
      val_cb = results_dict[key_cb]
      val_ce = results_dict[key_ce]
      log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
  print(log_str)
  results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
  print("Writing final ROUGE results to %s..."%(results_file))
  with open(results_file, "w") as f:
    f.write(log_str)


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
  if running_avg_loss == 0:  # on the first iteration just take the loss
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)  # clip
  loss_sum = tf.Summary()
  tag_name = 'running_avg_loss/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  return running_avg_loss
