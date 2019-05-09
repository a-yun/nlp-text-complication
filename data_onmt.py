from nltk.tokenize import wordpunct_tokenize
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split


def tok(x):
    return ' '.join(wordpunct_tokenize(x))


NEWSELA_FILE = './newsela_article_corpus_2016-01-29/' \
    'newsela_data_share-20150302/newsela_articles_20150302.aligned.sents.txt'

newsela_df = pd.read_csv(NEWSELA_FILE, sep='\t',
                         names=['doc_id', 'doc_version_complex',
                                'doc_version_simple', 'sentence_complex',
                                'sentence_simple']
                         )
'''
# Only keep sentences less than 1000 words in order to fit in CUDA memory
MAX_SENTENCE_LEN = 1000
newsela_df["simple_len"] = newsela_df['sentence_simple'].str.count(' ')
newsela_df["complex_len"] = newsela_df['sentence_complex'].str.count(' ')
newsela_df = newsela_df.query('simple_len < {len:d} & complex_len < {len:d}'
                              .format(len=MAX_SENTENCE_LEN))
newsela_df.drop(columns=["simple_len", "complex_len"], inplace=True)
'''

newsela_df.dropna(inplace=True)

# Filter out duplicate simple sentences to preserve train/val/test split
newsela_df.drop_duplicates(
    subset='sentence_complex',
    keep='first',
    inplace=True)

pd.set_option('display.max_colwidth', -1)

# Write to data files required by OpenNMT
pathlib.Path('/my/directory').mkdir(parents=True, exist_ok=True)

df_train, df_val = train_test_split(newsela_df, test_size=0.1, random_state=0)
df_train.to_csv(
    "../OpenNMT-py/data/src-train.txt",
    columns=['sentence_simple'],
    index=False,
    header=False)
df_train.to_csv(
    "../OpenNMT-py/data/trg-train.txt",
    columns=['sentence_complex'],
    index=False,
    header=False)
df_val.to_csv(
    "../OpenNMT-py/data/src-val.txt",
    columns=['sentence_simple'],
    index=False,
    header=False)
df_val.to_csv(
    "../OpenNMT-py/data/trg-val.txt",
    columns=['sentence_complex'],
    index=False,
    header=False)
