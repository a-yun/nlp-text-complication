from nltk.tokenize import wordpunct_tokenize
import pandas as pd

NEWSELA_FILE = './newsela_article_corpus_2016-01-29/newsela_data_share-20150302/newsela_articles_20150302.aligned.sents.txt'

newsela_df = pd.read_csv(NEWSELA_FILE, sep='\t',
                         names=['DOC-ID', 'DOC-VERSION-COMPLES',
                                'DOC-VERSION-SIMPLE', 'SENTENCE-COMPLEX',
                                'SENTENCE-SIMPLE']
                        )
newsela_df = newsela_df.loc[:, ["SENTENCE-SIMPLE", "SENTENCE-COMPLEX"]]
newsela_df = newsela_df.applymap(wordpunct_tokenize)
