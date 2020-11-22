import argparse
from data_loader import SemEvalProcessor, load_tokenizer, data_generator, load_config, \
    convert_examples_to_list, DataGenerator, load_albert_config, load_albert_tokenizer
from model_tf import RBERT, RALBERT
import tensorflow as tf
import os
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)


def train(args, is_albert=False, batch_size=16):
    semEvalProcessor = SemEvalProcessor(args)
    train_examples = semEvalProcessor.get_examples("train")
    test_examples = semEvalProcessor.get_examples("test")[0:1600]

    if is_albert:
        config = load_albert_config(args)
        rbert = RALBERT(config=config, args=args)
        tokenizer = load_albert_tokenizer()
    else:
        config = load_config(args)
        rbert = RBERT(config=config, args=args)
        tokenizer = load_tokenizer(args)

    print(config)

    train_input_ids, train_attention_mask, train_token_type_ids, train_e1_mask, train_e2_mask, train_label_ids \
        = convert_examples_to_list(train_examples, args.max_seq_len, tokenizer, cls_token="[CLS]", cls_token_segment_id=0,
                                   sep_token="[SEP]", pad_token=0, sequence_a_segment_id=0, add_sep_token=False, mask_padding_with_zero=True)
    trainDataGenerator = DataGenerator([train_input_ids, train_attention_mask, train_token_type_ids, train_e1_mask, train_e2_mask, train_label_ids])

    test_input_ids, test_attention_mask, test_token_type_ids, test_e1_mask, test_e2_mask, test_label_ids \
        = convert_examples_to_list(test_examples, args.max_seq_len, tokenizer, cls_token="[CLS]", cls_token_segment_id=0,
                                   sep_token="[SEP]", pad_token=0, sequence_a_segment_id=0, add_sep_token=False, mask_padding_with_zero=True)
    testDataGenerator = DataGenerator([test_input_ids, test_attention_mask, test_token_type_ids, test_e1_mask, test_e2_mask, test_label_ids], batch_size=32, is_test=True)
    callbacks_list = [
        tf.keras.callbacks.ReduceLROnPlateau(
            # This callback will monitor the validation loss of the model
            monitor='val_loss',
            # It will divide the learning by 10 when it gets triggered
            factor=0.7,
            # It will get triggered after the validation loss has stopped improving
            # for at least 10 epochs
            patience=4,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    ]

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1)
    loss = tf.keras.losses.sparse_categorical_crossentropy
    rbert.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    history = rbert.fit_generator(trainDataGenerator.__iter__(), steps_per_epoch=len(train_examples) // batch_size,
                                  validation_data=testDataGenerator.__iter__(), validation_steps=len(test_examples) // batch_size, epochs=50,
                                  shuffle=True, callbacks=callbacks_list)
    rbert.summary()
    save_model(rbert, "rbert.h5", args)
    return history


def save_model(model, model_name, args):
    model.save_weights(filepath=os.path.join(args.model_dir, model_name))


def train_steps(args):
    semEvalProcessor = SemEvalProcessor(args)
    train_examples = semEvalProcessor.get_examples("train")
    tokenizer = load_tokenizer(args)
    config = load_config(args)

    train_data_generator = data_generator(train_examples, args.max_seq_len, tokenizer,
                                          cls_token="[CLS]",
                                          cls_token_segment_id=0,
                                          sep_token="[SEP]", pad_token=0,
                                          pad_token_segment_id=0,
                                          sequence_a_segment_id=0,
                                          add_sep_token=False,
                                          mask_padding_with_zero=True, batch_size=2)

    rbert = RBERT(config=config, args=args)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss_fn = tf.keras.losses.sparse_categorical_crossentropy

    train_loss = []
    step = 0
    for epoch in range(100):
        for x, y in train_data_generator:
            with tf.GradientTape() as tape:
                y_pred = rbert(x)
                loss = loss_fn(y_true=y, y_pred=y_pred)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, rbert.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, rbert.variables))
            train_loss.append(loss.numpy())
            step += 1
            if step % 100 == 0:
                print("train loss:", sum(train_loss) / len(train_loss))
                train_loss = []


if __name__ == '__main__':
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="semeval", type=str, help="The name of the task to train")
    parser.add_argument("--dict_path", default=root_dir + "/data/vocab.txt")
    parser.add_argument(
        "--data_dir",
        default=root_dir + "/data/relation_extract/SemEval2010",
        type=str,
        help="The input SemEval2010 dir. Should contain the .tsv files (or other SemEval2010 files) for the task.",
    )
    parser.add_argument("--model_dir", default=root_dir + "/models/saved", type=str, help="Path to model")

    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Label file")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Model Name or Path",
    )
    parser.add_argument("--hidden_size", default=64, type=int, help="hidden size")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation.")
    parser.add_argument(
        "--max_seq_len",
        default=384,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )

    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout for fully-connected layers",
    )
    parser.add_argument("--seed", type=int, default=77, help="random seed for initialization")

    args = parser.parse_args()
    set_seed(args)
    train(args, is_albert=True)

    # semEvalProcessor = SemEvalProcessor(args)
    # train_examples = semEvalProcessor.get_examples("train")
    # tokenizer = load_tokenizer(args)
    # config = load_config(args)
    #
    # all_input_ids, all_attention_mask, all_token_type_ids, all_e1_mask, all_e2_mask, all_label_ids \
    #     = convert_examples_to_list(train_examples, args.max_seq_len, tokenizer, cls_token="[CLS]", cls_token_segment_id=0,
    #                                sep_token="[SEP]", pad_token=0, sequence_a_segment_id=0, add_sep_token=False, mask_padding_with_zero=True)
    # dataGenerator = DataGenerator([all_input_ids, all_attention_mask, all_token_type_ids, all_e1_mask, all_e2_mask, all_label_ids])
    # for x in dataGenerator.__iter__():
    #     print(x[0][0].shape, x[0][1].shape, x[0][2].shape, x[0][3].shape, x[0][4].shape, x[1].shape)
    #     break
    # train_data_generator = data_generator(test_examples, args.max_seq_len, tokenizer,
    #                                       cls_token="[CLS]",
    #                                       cls_token_segment_id=0,
    #                                       sep_token="[SEP]", pad_token=0,
    #                                       pad_token_segment_id=0,
    #                                       sequence_a_segment_id=0,
    #                                       add_sep_token=False,
    #                                       mask_padding_with_zero=True, batch_size=16)

    # rbert = RBERT(config=config, args=args)
    # for x in train_data_generator:
    #     print(x[0][0].shape, x[0][1].shape, x[0][2].shape, x[0][3].shape, x[0][4].shape, x[1].shape)
    #     print(x[1])
    #     # print(x[0][0])
    #     # print(x[0][1])
    #     # print(x[0][2])
    #     # print(x[0][3])
    #     # print(x[0][4])
    #     out = rbert([x[0][0], x[0][1], x[0][2], x[0][3], x[0][4]])
    #     print(rbert.summary())
    #     print(out.shape)
    #     break
    #     # print(out)
