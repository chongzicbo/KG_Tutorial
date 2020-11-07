from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input, Masking
from tensorflow.keras.models import Model
from config import *
from crf import CRF


def build_model():
    """
    使用Sequential构建模型网络:双向LSTM+CRF
    :return:
    """
    inputs = Input(shape=(None,), dtype='int32')
    output = Embedding(vocab_size, embedding_dim, trainable=True)(inputs)
    output = Masking(mask_value=0)(output)
    output = Bidirectional(LSTM(lstm_hidden_size, return_sequences=True))(output)
    output = Dropout(dropout_rate)(output)
    output = Bidirectional(LSTM(lstm_hidden_size, return_sequences=True))(output)
    output = Dropout(dropout_rate)(output)
    output = Dense(tag_size, activation=None)(output)
    crf = CRF(dtype="float32")
    output = crf(output)
    model = Model(inputs, output)
    model.compile(loss=crf.loss, optimizer=optimizer, metrics=[crf.accuracy])
    model.summary()
    return model


if __name__ == '__main__':
    model = build_model()
    train_x = np.array(padding_sequences_ids)
    train_y = np.array(padding_tags_ids)
    print(train_x.shape, train_y.shape)
    model.fit(x=train_x, y=train_y, epochs=1, batch_size=batch_size, validation_split=0.2)
