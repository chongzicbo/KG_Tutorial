from tensorflow import keras
from tensorflow.keras import layers
from config import MAX_LEN, BATCH_SIZE, LSTM_DECODE, LSTM_ENCODE, WORD_DICT, TAG_DICT, EMBEDDING_SIZE


def bilstm_lstm(word_size, tag_size):
    x = layers.Input(shape=MAX_LEN, batch_size=BATCH_SIZE)
    embedding_x = layers.Embedding(input_dim=word_size, output_dim=EMBEDDING_SIZE)(x)
    bilstm_encode = layers.Bidirectional(layers.LSTM(units=LSTM_ENCODE, return_sequences=True))(embedding_x)
    bilstm_decode = layers.LSTM(units=LSTM_DECODE, return_sequences=True)(bilstm_encode)
    out = layers.Dense(units=tag_size, activation="softmax")(bilstm_decode)
    model = keras.models.Model(x, out)
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    model.summary()
    return model


if __name__ == '__main__':
    bilstm_lstm()
