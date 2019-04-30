import argparse
import numpy as np
import os

import torch
import torchtext.vocab as vocab
from torch import nn, optim

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
    criterion = nn.CrossEntropyLoss().to(device)

    # TODO - pad correctly with loss function
    for epoch in range(num_epochs):
        total_loss = 0.0

        for idx_batch, batch in enumerate(train_iter):
            # Zero out the gradients from the model.
            model.zero_grad()

            probs = model.forward(
                batch.sentence_simple,
                batch.sentence_complex)
            loss = criterion(
                probs.permute(1, 2, 0),
                batch.sentence_complex.permute(1, 0))
            total_loss += loss

            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        print("Done with epoch")

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(dirname, 'model.pth'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--num_epochs', type=int, default=10000)
    parser.add_argument('-l', '--log_dir')
    args = parser.parse_args()

    print('[I] Start training')
    train(args.num_epochs, log_dir=args.log_dir)
    print('[I] Training finished')
