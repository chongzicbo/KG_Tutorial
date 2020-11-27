import os
# from data_helper import *

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
relation_file_path = root_dir + "/data/relation_extract/relation2id.txt"
train_file_path = root_dir + "/data/relation_extract/train.txt"
model_saved_path = root_dir + "/models/bilstm_attention_re.h5"

relation2id = get_relation2id(relation_file_path)
datas, labels, positionE1, positionE2 = get_sentence_label_positionE(train_file_path, relation2id)
word2id, id2word = get_word2id(datas)

datas, labels, positionE1, positionE2 = get_data_array(word2id, datas, labels, positionE1, positionE2)
EMBEDDING_SIZE = len(word2id) + 1
EMBEDDING_DIM = 100

POS_SIZE = 82  # 不同数据集这里可能会报错。
POS_DIM = 25

HIDDEN_DIM = 200

TAG_SIZE = len(relation2id)

BATCH = 64
EPOCHS = 100

config = {}
config['EMBEDDING_SIZE'] = EMBEDDING_SIZE
config['EMBEDDING_DIM'] = EMBEDDING_DIM
config['POS_SIZE'] = POS_SIZE
config['POS_DIM'] = POS_DIM
config['HIDDEN_DIM'] = HIDDEN_DIM
config['TAG_SIZE'] = TAG_SIZE
config['BATCH'] = BATCH
learning_rate = 0.0005
