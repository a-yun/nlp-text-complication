import argparse
import os

from nltk.translate.bleu_score import sentence_bleu
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

    pred_bleu = list()
    delta_bleu = list()
    baseline_bleu = list()

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

    print('Model-tgt BLEU score: ', sum(pred_bleu)/len(pred_bleu))
    print('Model-src BLEU score: ', sum(delta_bleu)/len(delta_bleu))
    print('Baseline BLEU score: ', sum(baseline_bleu)/len(baseline_bleu))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print('[I] Start eval')
    eval()
    print('[I] Eval finished')
