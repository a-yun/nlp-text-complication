from nltk.tokenize import wordpunct_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.data import Field, BucketIterator, TabularDataset

NEWSELA_FILE = './newsela_article_corpus_2016-01-29/newsela_data_share-20150302/newsela_articles_20150302.aligned.sents.txt'

newsela_df = pd.read_csv(NEWSELA_FILE, sep='\t',
                         names=['doc_id', 'doc_version_complex',
                                'doc_version_simple', 'sentence_complex',
                                'sentence_simple']
                        )

newsela_df.dropna(inplace=True)
train, val = train_test_split(newsela_df, test_size=0.1)

train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)

def tok(x):
    # print(x)
    # return map(wordpunct_tokenize, x)
    return wordpunct_tokenize(x)

SIMPLE_TEXT = Field(tokenize=tok)
COMPLEX_TEXT = Field(tokenize=tok)

data_fields = [('doc_id', None), ('doc_version_complex', None), ('doc_version_simple', None),
               ('sentence_complex', SIMPLE_TEXT), ('sentence_simple', COMPLEX_TEXT)]

train, val = TabularDataset.splits(path='./', train='train.csv', validation='val.csv',
                                   format='csv', fields=data_fields)

SIMPLE_TEXT.build_vocab(train, val)
COMPLEX_TEXT.build_vocab(train, val)

train_iter = BucketIterator(train, batch_size=20, shuffle=True,
                            sort_key=lambda x: len(x.sentence_complex))
