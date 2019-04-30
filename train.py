import argparse
import numpy as np
import os

import torch
import torchtext.vocab as vocab
from torch import nn, optim

from models import Seq2seq

dirname = os.path.dirname(os.path.abspath(__file__))


def train(num_epochs, batch_size=1, lr=0.001, log_dir=None):
    '''
    TODO - comment
    '''
    # Load GloVe vectors and initialize model
    glove = vocab.GloVe(name='6B', dim=100)
    model = Seq2seq(glove, 200)

    # Load the training data
    # TODO - this is dummy data
    # TODO - add special tokens
    train_inputs = [['i', 'like', 'pie']]
    train_inputs = [[glove.stoi[w] for w in input] for input in train_inputs]
    train_inputs = torch.tensor(train_inputs)

    train_labels = [['one', 'appreciates', 'pastry']]
    train_labels = [[glove.stoi[w] for w in input] for input in train_labels]
    train_labels = torch.tensor(train_labels)
    #train_inputs, train_labels = load(os.path.join('tux_train.dat'))
    #val_inputs, val_labels = load(os.path.join('tux_valid.dat'))

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(0, num_epochs):
        total_loss = 0.0
        ex_indices = list(range(0, len(train_inputs)))
        # random.shuffle(ex_indices)

        for idx in ex_indices:
            x = train_inputs[idx].unsqueeze(1)
            y = train_labels[idx].unsqueeze(1)

            # Zero out the gradients from the model.
            model.zero_grad()
            probs = model.forward(x, y)[:, -1]
            loss = criterion(probs, y.squeeze())
            total_loss += loss

            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()

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
