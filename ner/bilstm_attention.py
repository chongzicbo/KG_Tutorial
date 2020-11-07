import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Input, Attention
from config import *


def build_model():
    """
    使用Sequential构建模型网络:双向LSTM+self-attention
    :return:
    """
    query_input = Input(shape=(None,), dtype="int32")
    token_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    query_embeddings = token_embedding(query_input)
    value_embedding = token_embedding(query_input)
    bilstm = Bidirectional(LSTM(lstm_hidden_size, return_sequences=True))

    query_seq_encoding = bilstm(query_embeddings)
    value_seq_encoding = bilstm(value_embedding)

    attention = Attention()([query_seq_encoding, value_seq_encoding])
    output = Dense(tag_size, activation="softmax")(attention)
    model = Model(query_input, output)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
    model.summary()
    return model


if __name__ == '__main__':
    model = build_model()
    train_x = np.array(padding_sequences_ids)
    train_y = np.array(padding_tags_ids)
    print(train_x.shape, train_y.shape)
    model.fit(x=train_x, y=train_y, epochs=1, batch_size=batch_size, validation_split=0.2)
