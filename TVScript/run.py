import helper
import numpy as np
import problem_unittests as tests
import Preprocess
import TVScriptGen
import params
import torch

train_on_gpu = params.get_device()
text = helper.load_data(params.data_dir)
view_line_range = (0, 10)
lines = text.split('\n')
word_count_line = [len(line.split()) for line in lines]
if params.log:
    print('Dataset Stats')
    print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
    print('Number of lines: {}'.format(len(lines)))
    print('Average number of words in each line: {}'.format(np.average(word_count_line)))
    print()
    print('The lines {} to {}:'.format(*view_line_range))
    print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))
words = text.split()
vocab_to_int, int_to_vocab = Preprocess.create_lookup_tables(words)
int_text = [vocab_to_int[word] for word in words]
Preprocess.helper_preprocess()
if params.load_settings:
    int_text, vocab_to_int, int_to_vocab, token_dict = Preprocess.helper_load_preprocess_settings()
else:
    token_dict = None
# pre-process training data
# int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
# Test Batching
if params.log:
    test_text = range(50)
    t_loader = TVScriptGen.batch_data(test_text, sequence_length=5, batch_size=10)
    data_iter = iter(t_loader)
    sample_x, sample_y = data_iter.next()
    print(sample_x.shape)
    print(sample_x)
    print()
    print(sample_y.shape)
    print(sample_y)
    print()
data_loader = TVScriptGen.batch_data(int_text, params.seq_length, params.batch_size)
tests.test_rnn(TVScriptGen.RNN, train_on_gpu)
tests.test_forward_back_prop(TVScriptGen.RNN, TVScriptGen.forward_back_prop, train_on_gpu)
vocab_size = len(vocab_to_int)
output_size = len(vocab_to_int)
rnn = TVScriptGen.RNN(vocab_size, output_size, params.embedding_dim, params.hidden_dim, params.n_layers, dropout=0.5,
                      train_on_gpu=train_on_gpu)
# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=params.learning_rate)
criterion = torch.nn.CrossEntropyLoss()
# training the model
trained_rnn = \
    TVScriptGen.train_rnn(rnn, params.batch_size, optimizer, criterion, params.num_epochs,
                          data_loader, params.show_every_n_batches, train_on_gpu)
# saving the trained model
helper.save_model('./save/trained_rnn', trained_rnn)
print('Model Trained and Saved')
_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
trained_rnn = helper.load_model('./save/trained_rnn')
gen_length = 400 # modify the length to your preference
prime_word = 'jerry' # name for starting the script

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
pad_word = helper.SPECIAL_WORDS['PADDING']
generated_script = \
    TVScriptGen.generate(trained_rnn, vocab_to_int[prime_word + ':'], int_to_vocab,
                         token_dict, vocab_to_int[pad_word], gen_length)
print(generated_script)

