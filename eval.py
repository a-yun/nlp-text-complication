import argparse
import os

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize.treebank import TreebankWordDetokenizer
from textstat import flesch_kincaid_grade
import scipy
import torch
from torch import nn, optim
from tqdm import tqdm

# TODO - put this into a function later
from data import *
from models import Seq2seq

dirname = os.path.dirname(os.path.abspath(__file__))


def eval(batch_size=1):
    '''
    TODO - comment
    '''
    model = Seq2seq(SIMPLE_TEXT.vocab, 200)
    model.to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(
                dirname,
                'model.pth'),
            map_location=device))
    model.eval()

    d = TreebankWordDetokenizer()
    pred_bleu = list()
    delta_bleu = list()
    baseline_bleu = list()

    pred_fk = list()
    simple_fk = list()
    complex_fk = list()

    pred_len = list()
    simple_len = list()
    complex_len = list()

    with torch.no_grad():
        for batch in tqdm(iter(test_iter), total=len(test_iter)):
            for simple, complex in zip(
                    batch.sentence_simple[0].permute(1, 0),
                    batch.sentence_complex[0].permute(1, 0)):
                pred, _ = model.translate_greedy(simple.unsqueeze(1))
                simple_text = [SIMPLE_TEXT.vocab.itos[tok] for tok in simple]
                complex_text = [SIMPLE_TEXT.vocab.itos[tok] for tok in complex]
                pred_bleu.append(sentence_bleu([complex_text], pred))
                delta_bleu.append(sentence_bleu([simple_text], pred))
                baseline_bleu.append(sentence_bleu([complex_text], simple_text))

                pred_fk.append(flesch_kincaid_grade(d.detokenize(pred)))
                simple_fk.append(flesch_kincaid_grade(d.detokenize(simple_text)))
                complex_fk.append(flesch_kincaid_grade(d.detokenize(complex_text)))

                pred_len.append(len(d.detokenize(pred)))
                simple_len.append(len(d.detokenize(simple_text)))
                complex_len.append(len(d.detokenize(complex_text)))

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print('[I] Start eval')
    eval()
    print('[I] Eval finished')
