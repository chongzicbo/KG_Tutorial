from data_helper import *
import os

embedding_dim = 128
batch_size = 64
lstm_hidden_size = 64
dropout_rate = 0.4
optimizer = "adam"
metrics = "accuracy"
loss = "sparse_categorical_crossentropy"

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
train_path = root_dir + "/data/ner/train.txt"
vocab_path = root_dir + "/data/vocab.txt"
w2i, i2w = get_w2i(vocab_path)
index2tag, tag2index, sequences, tags_list, max_len = get_tag2index(train_path)
sequences_ids, padding_sequences_ids, tags_ids, padding_tags_ids = get_data_array(sequences, tags_list, w2i, tag2index,
                                                                                  max_len)

vocab_size = len(w2i)  # 单词数量
tag_size = len(index2tag)  # 标签数量

# albert预训练模型文件
config_path = root_dir + '/models/pretrained/albert_tiny_zh_google/albert_config.json'
checkpoint_path = root_dir + '/models/pretrained/albert_tiny_zh_google/albert_model.ckpt'
dict_path = root_dir + '/models/pretrained/albert_tiny_zh_google/vocab.txt'
bert_sequence_ids, bert_datatype_ids, bert_label_ids = get_data_array_bert(padding_sequences_ids, padding_tags_ids, tag2index)
