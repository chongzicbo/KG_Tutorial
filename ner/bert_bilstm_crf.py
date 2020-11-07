import os

os.environ['TF_KERAS'] = '1'  # 必须放在前面,才能使用tf.keras
from bert4keras.models import build_transformer_model
from config import *
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.models import Model
from crf import CRF


def build_model(use_bilstm=True, use_crf=True):
    albert = build_transformer_model(config_path, checkpoint_path, model='albert', return_keras_model=False)  # 建立模型，加载权重
    output = albert.model.output
    if use_bilstm:
        output = Bidirectional(LSTM(lstm_hidden_size, return_sequences=True))(output)
        output = Dropout(dropout_rate)(output)
    if use_crf:
        activation = None
    else:
        activation = "softmax"
    output = Dense(tag_size, activation=activation, kernel_initializer=albert.initializer)(output)
    if use_crf:
        crf = CRF(dtype="float32")
        output = crf(output)
    model = Model(albert.model.inputs, output)
    model.compile(optimizer=optimizer, loss=crf.loss, metrics=[crf.accuracy])
    model.summary()
    return model


if __name__ == '__main__':
    model = build_model()
    train_x1 = np.array(bert_sequence_ids)
    train_x2 = np.array(bert_datatype_ids)
    train_y = np.array(bert_label_ids)
    print(train_x1.shape,train_x2.shape, train_y.shape)
    # model.fit(x=[train_x1, train_x2], y=train_y, epochs=10, batch_size=8, validation_split=0.2)
