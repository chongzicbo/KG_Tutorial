import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Model

"""
参考论文
Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
"""


class BiLSTMAttention(Model):
    def __init__(self, config: dict):
        super(BiLSTMAttention, self).__init__()
        self.batch = config["BATCH"]
        self.embedding_size = config["EMBEDDING_SIZE"]
        self.embedding_dim = config["EMBEDDING_DIM"]
        self.hidden_dim = config["HIDDEN_DIM"]
        self.tag_size = config["TAG_SIZE"]
        self.pos_size = config["POS_SIZE"]
        self.pos_dim = config["POS_DIM"]

        self.word_embeds = Embedding(self.embedding_size, self.embedding_dim)
        self.pos1_embeds = Embedding(self.pos_size, self.pos_dim)
        self.pos2_embeds = Embedding(self.pos_size, self.pos_dim)
        self.bilstm = Bidirectional(LSTM(self.hidden_dim // 2, return_sequences=True))
        self.dense = Dense(self.tag_size, activation="softmax")
        self.dropout_lstm = Dropout(0.5)
        self.drop_att = Dropout(0.5)
        self.att_weight = tf.Variable(tf.random.normal(shape=(self.batch, 1, self.hidden_dim)))
        self.relation_bias = tf.Variable(tf.random.normal(shape=(self.batch, self.tag_size, 1)))

    def attention(self, H):
        M = tf.tanh(H)
        a = tf.nn.softmax(tf.matmul(self.att_weight, M), 2)
        a = tf.transpose(a, perm=[0, 2, 1])
        return tf.matmul(H, a)

    def call(self, inputs, training=True):
        embeds = tf.concat((self.word_embeds(inputs[0]), self.pos1_embeds(inputs[1]),
                            self.pos2_embeds(inputs[2])), axis=2)
        # print("embeds shape:", embeds.shape)
        bilstm_out = self.bilstm(embeds)
        # print("lstm_out shape:", bilstm_out.shape)
        if training:
            bilstm_out = self.dropout_lstm(bilstm_out)
        bilstm_out = tf.transpose(bilstm_out, perm=[0, 2, 1])
        # print("transpose lstm_out shape:", bilstm_out.shape)
        att_out = tf.tanh(self.attention(bilstm_out))
        # print("attn_out:", att_out.shape)
        if training:
            att_out = self.drop_att(att_out)
        res = self.dense(tf.squeeze(att_out))
        # print("res shape", res.shape)
        return res


if __name__ == '__main__':
    EMBEDDING_SIZE = 100
    EMBEDDING_DIM = 100

    POS_SIZE = 82  # 不同数据集这里可能会报错。
    POS_DIM = 25

    HIDDEN_DIM = 200

    TAG_SIZE = 12

    BATCH = 128
    EPOCHS = 100

    config = {}
    config['EMBEDDING_SIZE'] = EMBEDDING_SIZE
    config['EMBEDDING_DIM'] = EMBEDDING_DIM
    config['POS_SIZE'] = POS_SIZE
    config['POS_DIM'] = POS_DIM
    config['HIDDEN_DIM'] = HIDDEN_DIM
    config['TAG_SIZE'] = TAG_SIZE
    config['BATCH'] = BATCH
    config["pretrained"] = False

    learning_rate = 0.0005
    model = BiLSTMAttention(config)
    sentence = tf.ones(shape=(BATCH, 50), dtype=tf.int32)
    pos1 = tf.ones(shape=(BATCH, 50), dtype=tf.int32)
    pos2 = tf.ones(shape=(BATCH, 50), dtype=tf.int32)
    model([sentence, pos1, pos2])
    model.summary()
