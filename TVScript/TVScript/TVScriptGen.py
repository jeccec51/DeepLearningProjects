import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import helper
import problem_unittests
import numpy as np
import params
import torch.nn.functional as F


# _, train_on_gpu = params.get_device()


def batch_data(words, sequence_length, batch_size, log=1):
    """
    Batch the neural network data using DataLoader
    :param log: Logs the examples
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: data_loaders: DataLoader with batched data
    """
    n_batches = len(words) // batch_size
    # Filter the residual data so that we have only full atches
    words = words[:n_batches * batch_size]
    future_tensors, targets = [], []
    for ii in range(0, len(words) - sequence_length):
        start = ii
        end = sequence_length + ii
        data = words[start:end]
        future_tensors.append(data)
        if (end + 1) >= (len(words)):
            target_word = words[0]
        else:
            target_word = words[end + 1]
        targets.append(target_word)
    # create Tensor datasets
    data = TensorDataset(torch.from_numpy(np.asarray(future_tensors)), torch.from_numpy(np.asarray(targets)))
    data_loader = DataLoader(data, shuffle=False, batch_size=batch_size)
    return data_loader


class RNN(nn.Module):
    def __init__(self, vocab_size, output_size, embed_dim, hidden_dim, n_layers, dropout=0.5, train_on_gpu=False):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embed_dim: The size of embeddings, should you choose to use them
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # Internal Variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.train_on_gpu = train_on_gpu
        # Layers
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            dropout=dropout, batch_first=True)
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        # linear  layers
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        batch_size = nn_input.size(0)
        # embeddings and lstm_out
        # nn_input = nn_input.long()
        embeds = self.embedding(nn_input)
        lstm_out, hidden = self.lstm(embeds, hidden)
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # reshape into (batch_size, seq_length, output_size)
        out = out.view(batch_size, -1, self.output_size)
        out = out[:, -1]  # get last batch of labels
        # return one batch of output word scores and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if self.train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden


def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden, train_on_gpu=False):
    """
    Forward and backward propagation on the neural network
    :param train_on_gpu: Flag to indicate of GPU is available
    :param hidden: Hidden state output
    :param rnn: The PyTorch Module that holds the neural network
    :param optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """

    # TODO: Implement Function

    # move data to GPU, if available

    # perform backpropagation and optimization

    # return the loss over a batch and the hidden state produced by our model
    clip = 5  # gradient clipping

    # Creating new variables for the hidden state
    hidden = tuple([each.data for each in hidden])
    # move data to GPU, if available
    if train_on_gpu:
        inp, target = inp.cuda(), target.cuda()
    # zero accumulated gradients
    rnn.zero_grad()
    # perform backpropagation and optimization
    output, hidden = rnn(inp, hidden)
    loss = criterion(output.squeeze(), target)
    loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    nn.utils.clip_grad_norm_(rnn.parameters(), clip)
    optimizer.step()
    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden


def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, train_loader, show_every_n_batches=100,
              train_on_gpu=False):
    if train_on_gpu:
        rnn = rnn.cuda()
    batch_losses = []
    rnn.train()
    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset) // batch_size
            if batch_i > n_batches:
                break
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden, train_on_gpu)
            # record loss
            batch_losses.append(loss)
            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []
    # returns a trained rnn
    return rnn


def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
    """
    Generate text using the neural network
    :param rnn: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()

    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, params.seq_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]

    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)

        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))

        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)

        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if train_on_gpu:
            p = p.cpu()  # move to cpu

        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()

        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p / p.sum())

        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)

        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i

    gen_sentences = ' '.join(predicted)

    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')

    # return all the sentences
    return gen_sentences
