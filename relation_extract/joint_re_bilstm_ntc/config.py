import os

root_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                        "data/relation_extract/NYT/")
TRAIN_PATH = root_dir + 'train.json'
TEST_PATH = root_dir + 'test.json'
WORD_DICT = root_dir + 'word_dict.txt'
TAG_DICT = root_dir + 'tag_dict.txt'

MAX_LEN = 200
EMBEDDING_SIZE = 300
DROPOUT = 0.5
LSTM_ENCODE = 300
LSTM_DECODE = 600
EPOCH_NUM = 1000
BATCH_SIZE = 16
