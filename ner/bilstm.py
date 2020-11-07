from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, Masking
from tensorflow.keras.models import Sequential
from config import *
import os


def build_model():
    """
    使用Sequential构建模型网络:双向LSTM
    :return:
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(Masking(mask_value=0))
    model.add(Bidirectional(LSTM(lstm_hidden_size, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(lstm_hidden_size, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(tag_size, activation="softmax"))
    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
    model.summary()
    return model


if __name__ == '__main__':
    model = build_model()
    train_x = np.array(padding_sequences_ids)
    train_y = np.array(padding_tags_ids)
    # model.fit(x=train_x, y=train_y, epochs=1, batch_size=batch_size, validation_split=0.2)
