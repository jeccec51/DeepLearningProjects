import helper
import numpy as np
import problem_unittests as tests
import Preprocess

data_dir = './data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)
view_line_range = (0, 10)

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))

vocab_to_int, int_to_vocab = Preprocess.create_lookup_tables(text.split())
# pre-process training data
helper.preprocess_and_save_data(data_dir, Preprocess.token_lookup, Preprocess.create_lookup_tables)


v =1