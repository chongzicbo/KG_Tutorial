from data_helper import get_data, word_tag_dict, DataGenerator
from config import MAX_LEN, TRAIN_PATH, TEST_PATH, WORD_DICT, TAG_DICT, EPOCH_NUM, BATCH_SIZE, model_save_path
from model import bilstm_lstm


def get_class_weight(tag_dict):
    """
        不同标签的权重不一样，标签"O"的权重为1，其他的为10
    :param tag_dict:
    :return:
    """
    class_weight = {}
    for tag, index in tag_dict.items():
        if tag == "O":
            class_weight[index] = 1
        else:
            class_weight[index] = 10
    return class_weight


def train_bilstm_lstm():
    x_train, y_train, x_test, y_test = get_data(TRAIN_PATH, TEST_PATH)
    word_dict, tag_dict = word_tag_dict(WORD_DICT, TAG_DICT)
    train_dataGenerator = DataGenerator(word_dict, tag_dict, x_train, y_train, BATCH_SIZE, MAX_LEN)
    test_dataGenerator = DataGenerator(word_dict, tag_dict, x_test, y_test, BATCH_SIZE, MAX_LEN)
    class_weight = get_class_weight(tag_dict)
    print(class_weight)
    model = bilstm_lstm(len(word_dict) + 1, len(tag_dict))
    model.fit_generator(train_dataGenerator.__iter__(), epochs=EPOCH_NUM, steps_per_epoch=train_dataGenerator.steps,
                        validation_data=test_dataGenerator, validation_steps=test_dataGenerator.steps, class_weight=class_weight)
    model.save_weights(filepath=model_save_path)


if __name__ == '__main__':
    train_bilstm_lstm()
