import argparse
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize.treebank import TreebankWordDetokenizer
from textstat import flesch_kincaid_grade
import scipy

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--simple_file')
parser.add_argument('-c', '--complex_file')
parser.add_argument('-p', '--pred_file')
args = parser.parse_args()

simple_lines = list()
complex_lines = list()
pred_lines = list()

with open(args.simple_file) as f:
    simple_lines = [l.lower().split(' ') for l in list(f)]
with open(args.complex_file) as f:
    complex_lines = [l.lower().split(' ') for l in list(f)]
with open(args.pred_file) as f:
    pred_lines = [l.lower().split(' ') for l in list(f)]

pred_bleu = [sentence_bleu([c], p) for c, p in zip(complex_lines, pred_lines)]
delta_bleu = [sentence_bleu([s], p) for s, p in zip(simple_lines, pred_lines)]
baseline_bleu = [sentence_bleu([c], s) for c, s in zip(complex_lines, simple_lines)]

d = TreebankWordDetokenizer()
pred_fk = [flesch_kincaid_grade(d.detokenize(p)) for p in pred_lines]
simple_fk = [flesch_kincaid_grade(d.detokenize(s)) for s in simple_lines]
complex_fk = [flesch_kincaid_grade(d.detokenize(c)) for c in complex_lines]

pred_len = [len(d.detokenize(p)) for p in pred_lines]
simple_len = [len(d.detokenize(s)) for s in simple_lines]
complex_len = [len(d.detokenize(c)) for c in complex_lines]

print('Model-tgt BLEU score: ', sum(pred_bleu)/len(pred_bleu))
print('Model-src BLEU score: ', sum(delta_bleu)/len(delta_bleu))
print('Baseline BLEU score: ', sum(baseline_bleu)/len(baseline_bleu))
print()

print('Model FK grade level: ', sum(pred_fk)/len(pred_fk))
print('Simple FK grade level: ', sum(simple_fk)/len(simple_fk))
print('Complex FK grade level: ', sum(complex_fk)/len(complex_fk))
print('Simple-Model FK test: ', scipy.stats.ttest_rel(pred_fk, simple_fk))
print('Complex-Model FK test: ', scipy.stats.ttest_rel(complex_fk, simple_fk))
print()

print('Model sentence length: ', sum(pred_len)/len(pred_len))
print('Simple sentence length: ', sum(simple_len)/len(simple_len))
print('Complex sentence length: ', sum(complex_len)/len(complex_len))
print('Simple-Model len test: ', scipy.stats.ttest_rel(pred_len, simple_len))
print('Complex-Model len test: ', scipy.stats.ttest_rel(complex_len, simple_len))
print()
