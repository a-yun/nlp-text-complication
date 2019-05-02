import argparse
import os

import torch
from torch import nn, optim
from tqdm import tqdm

# TODO - put this into a function later
from data import *
from models import Seq2seq

dirname = os.path.dirname(os.path.abspath(__file__))


def train(num_epochs, batch_size=1, lr=0.001, log_dir=None):
    '''
    TODO - comment
    '''
    # TODO - move this into data.py
    # avg_emb = glove.vectors.mean(dim=0)
    model = Seq2seq(SIMPLE_TEXT.vocab, 200)
    model.to(device)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pad_idx = SIMPLE_TEXT.vocab.stoi[SIMPLE_TEXT.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)

    # TODO - pad correctly with loss function
    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in tqdm(iter(train_iter), total=len(train_iter)):
            probs = model.forward(
                batch.sentence_simple[0],
                batch.sentence_complex[0],
                batch.sentence_simple[1])
            loss = criterion(
                probs.permute(1, 2, 0),
                batch.sentence_complex[0].permute(1, 0))

            # Zeroes and omputes the gradient and takes the optimizer step
            model.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # This is necessary for some reason
            # if device.type == 'cuda':
                # torch.cuda.empty_cache()

        print("Done with epoch. Total loss:", total_loss)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(dirname, 'model.pth'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--num_epochs', type=int, default=16)
    parser.add_argument('-l', '--log_dir')
    args = parser.parse_args()

    print('[I] Start training')
    train(args.num_epochs, log_dir=args.log_dir)
    print('[I] Training finished')
