import torch
import torch.nn.functional as F
from torch import nn


class Seq2seq(nn.Module):
    '''
    This model implements the sequence to sequence model using RNNs.
    '''

    def __init__(self, vocab, hidden_size):
        super().__init__()
        '''
        Model initialization code.

        :param vocab: torchtext.vocab.Vocab pretrained embeddings
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
        Runs RNN on input word sequence to find probability of output sequence
        using teacher forcing.

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
        Runs RNN on input word sequence to predict the output translation
        using greedy search.

        :param src: a sequence of word vectors for the source text
        :return pred: a sequence of translated words in the target language
        :return out: real-valued scores for each word in the vocabulary
        '''
        src_emb = self.embedding(src)
        enc_out, (enc_hn, enc_cn) = self.encoder(src_emb)

        preds = ['<sos>']
        probs = [torch.Tensor(1, 1, len(self.vocab.itos))]
        probs[0][0, 0, self.vocab.stoi[preds[-1]]] = 1  # set <sos> prob to 1
        hn, cn = enc_hn, enc_cn

        # Predict one word at a time until EOS, reusing previous hidden states
        while preds[-1] != '<eos>' and len(preds) < 400:
            pred_idx = self.vocab.stoi[preds[-1]]
            trg_emb = self.embedding(torch.LongTensor([pred_idx])).unsqueeze(0)
            dec_out, (hn, cn) = self.decoder(trg_emb, (hn, cn))
            out = self.fc(dec_out)
            # print(out)
            # print(out.argmax(dim=-1))

            preds.append(self.vocab.itos[out.argmax(dim=-1)])
            probs.append(out)

        probs = torch.stack(probs, dim=0)
        return preds, probs

    def nearest_neighbor(self, vec):
        '''
        Find closest word to the given vector in the embedding space.
        https://github.com/spro/practical-pytorch/blob/master/glove-word-vectors/glove-word-vectors.ipynb

        :param vec: embedding-dimensional vector
        '''
        cos_sim = [(w, F.cosine_similarity(vec, self.vocab.vectors[i]))
                   for w, i in self.vocab.stoi.items()]
        return sorted(cos_sim, key=lambda t: t[1])[-1]
