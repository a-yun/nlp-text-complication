import torch
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
                               num_layers=1,
                               bidirectional=True)
        self.decoder = nn.LSTM(input_size=emb_dim,
                               hidden_size=hidden_size * 2,
                               num_layers=1,
                               bidirectional=False)

        self.activation = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, src, trg, src_lens):
        '''
        Runs RNN on input word sequence to find probability of output sequence
        using teacher forcing.

        :param src: a sequence of word embedding indices in the source text
        :param trg: a sequence of word embedding indices in the target text
        :param src_lens: lengths of each unpadded input seq, used for packing
        :return: real-valued scores for each word in the vocabulary
        '''
        src_emb = self.embedding(src)
        trg_emb = self.embedding(trg)

        # Pack padded tensors
        src_emb = torch.nn.utils.rnn.pack_padded_sequence(src_emb, src_lens)

        # RNN encoder
        enc_out, (enc_hn, enc_cn) = self.encoder(src_emb)

        # Resize bidirectional encoder state to fit unidirectional decoder
        # [dirs, batch, hid] -> [1, batch, hid * dirs]
        enc_hn = enc_hn.permute(1, 0, 2)  # switch to [batch, dirs, hid]
        enc_hn = enc_hn.contiguous()
        enc_hn = enc_hn.view(enc_hn.size()[0], 1, -1)  # [batch, 1, hid * dirs]
        enc_hn = enc_hn.permute(1, 0, 2)  # [1, batch, hid * dirs]
        enc_cn = enc_cn.permute(1, 0, 2)  # switch to [batch, dirs, hid]
        enc_cn = enc_cn.contiguous()
        enc_cn = enc_cn.view(enc_cn.size()[0], 1, -1)  # [batch, 1, hid * dirs]
        enc_cn = enc_cn.permute(1, 0, 2)  # [1, batch, hid * dirs]

        # RNN decoder
        dec_out, (dec_hn, dec_cn) = self.decoder(trg_emb, (enc_hn, enc_cn))

        # Undo packing (not necessary?)
        # dec_out = torch.nn.utils.rnn.pad_packed_sequence(dec_out)

        # Output logits for each word
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
        probs = [torch.Tensor(1, 1, len(self.vocab.itos)).to(device)]
        probs[0][0, 0, self.vocab.stoi[preds[-1]]] = 1  # set <sos> prob to 1
        hn, cn = enc_hn, enc_cn

        # Predict one word at a time until EOS, reusing previous hidden states
        while preds[-1] != '<eos>' and len(preds) < 400:
            pred_idx = self.vocab.stoi[preds[-1]]
            trg_emb = self.embedding(
                torch.LongTensor(
                    [pred_idx]).to(device)).unsqueeze(0)
            # TODO - fix this for biLSTM
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
