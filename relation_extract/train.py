from config import datas, labels, positionE1, positionE2, config, EPOCHS
from bilstm_attention_tf import BiLSTMAttention


def train():
    model = BiLSTMAttention(config)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x=[datas, positionE1, positionE2], y=labels, batch_size=config["BATCH"], epochs=EPOCHS, validation_split=0.2)
    model.summary()
    return history


if __name__ == '__main__':
    train()
