import torch
import torch.nn.functional as F
from torch import nn

# Unlike previous classes, don't use these classes directly in train.py
# Use the functions given in main.py
# However, model definition still needs to be defined in these classes


class Seq2seq(nn.Module):
    '''
    This model implements the sequence to sequence model using RNNs.
    '''

    def __init__(self, vocab, hidden_size):
        super().__init__()
        '''
        Model initialization code.

        :param vocab: torchtext.vocab.Vocab pretrained embeddings with special tokens added
        :param hidden_size: size of RNN hidden layer
        '''
        self.vocab = vocab
        vocab_size = len(vocab.itos)
        emb_dim = vocab.vectors.size()[1]

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight.data.copy_(vocab.vectors)
        self.encoder = nn.LSTM(input_size=emb_dim,
                               hidden_size=hidden_size,
                               num_layers=1)
        self.decoder = nn.LSTM(input_size=emb_dim,
                               hidden_size=hidden_size,
                               num_layers=1)

        self.activation = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, trg):
        '''
        Runs RNN on input word sequence to find probability of output sequence using teacher forcing.

        :param src: a sequence of word embedding indices in the source text
        :param trg: a sequence of word embedding indices in the target text
        :return: real-valued scores for each word in the vocabulary
        '''
        src_emb = self.embedding(src)
        trg_emb = self.embedding(trg)

        enc_out, (enc_hn, enc_cn) = self.encoder(src_emb)
        dec_out, (dec_hn, dec_cn) = self.decoder(trg_emb, (enc_hn, enc_cn))
        out = self.fc(dec_out)
        return out

    def translate_greedy(self, src):
        '''
        Runs RNN on input word sequence to predict the output translation using greedy search.

        :param src: a sequence of word vectors for the source text
        :return: real-valued scores for each word in the vocabulary
        '''
        enc_out, (enc_hn, enc_cn) = self.encoder(src)

        pred = ['<SOS>']
        hn, cn = enc_hn, enc_cn
        while pred[-1] != '<EOS>':
            dec_out, (hn, cn) = self.decoder(trg, (enc_hn, None))
            pred.append(self.nearest_neighbor(dec_out))
        out = self.fc(dec_out)
        return out

    def nearest_neighbor(self, vec):
        '''
        Find closest word to the given vector in the embedding space.
        https://github.com/spro/practical-pytorch/blob/master/glove-word-vectors/glove-word-vectors.ipynb

        :param vec: embedding-dimensional vector
        '''
        cos_sim = [(w, F.cosine_similarity(vec, vocab.vectors[i]))
                   for w, i in vocab.stoi.items()]
        return sorted(cos_sim, key=lambda t: t[1])[:n]
