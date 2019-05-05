from nltk.tokenize import wordpunct_tokenize
import pandas as pd
import random
import torch
from torchtext.data import Field, BucketIterator, TabularDataset

# From: https://stackoverflow.com/a/53374933
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')


NEWSELA_FILE = './newsela_article_corpus_2016-01-29/' \
    'newsela_data_share-20150302/newsela_articles_20150302.aligned.sents.txt'

newsela_df = pd.read_csv(NEWSELA_FILE, sep='\t',
                         names=['doc_id', 'doc_version_complex',
                                'doc_version_simple', 'sentence_complex',
                                'sentence_simple']
                         )

newsela_df["simple_len"] = newsela_df['sentence_simple'].str.count(' ')
newsela_df["complex_len"] = newsela_df['sentence_complex'].str.count(' ')

MAX_SENTENCE_LEN = 400
newsela_df = newsela_df.query('simple_len < {len:d} & complex_len < {len:d}'
                              .format(len=MAX_SENTENCE_LEN))

newsela_df.drop(columns=["simple_len", "complex_len"], inplace=True)
newsela_df.dropna(inplace=True)
newsela_df.to_csv("data.csv", index=False)


def tok(x):
    return wordpunct_tokenize(x)


SIMPLE_TEXT = Field(tokenize=tok, include_lengths=True,
                    init_token='<sos>', eos_token='<eos>')
COMPLEX_TEXT = Field(tokenize=tok, include_lengths=True,
                     init_token='<sos>', eos_token='<eos>')

data_fields = [
    ('doc_id', None),
    ('doc_version_complex', None),
    ('doc_version_simple', None),
    ('sentence_complex', SIMPLE_TEXT),
    ('sentence_simple', COMPLEX_TEXT)]

dataset = TabularDataset(path='./data.csv', format='csv', fields=data_fields)
train, val, test = dataset.split(
    split_ratio=[0.8, 0.1, 0.1],
    random_state=random.seed(0))

# TODO - average unk_init
SIMPLE_TEXT.build_vocab(
    train,
    val,
    test,
    vectors="glove.6B.100d",
    unk_init=None)
COMPLEX_TEXT.vocab = SIMPLE_TEXT.vocab

BATCH_SIZE = 32
train_iter, val_iter, test_iter = BucketIterator.splits(
    (train, val, test), device=device,
    batch_sizes=(BATCH_SIZE, BATCH_SIZE, BATCH_SIZE),
    shuffle=True, sort_within_batch=True,
    sort_key=lambda x: len(x.sentence_simple))
