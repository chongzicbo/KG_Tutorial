import copy
import csv
import json
import os

import numpy as np

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

from transformers import BertTokenizer, BertConfig, AlbertTokenizer, AlbertConfig


def get_label(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), "r", encoding="utf-8")]


def load_config(args):
    label_lst = get_label(args)
    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_lst),
        finetuning_task=args.task,
        id2label={str(i): label for i, label in enumerate(label_lst)},
        label2id={label: i for i, label in enumerate(label_lst)},
    )
    # config.hidden_size = 384
    # config.intermediate_size = 768
    # config.num_hidden_layers = 6
    # config.num_attention_heads = 6
    # config.hidden_dropout_prob = args.drop_rate
    return config


def load_albert_config(args):
    label_lst = get_label(args)
    config = AlbertConfig.from_pretrained("albert-large-v2", num_labels=len(label_lst),
                                          finetuning_task=args.task,
                                          id2label={str(i): label for i, label in enumerate(label_lst)},
                                          label2id={label: i for i, label in enumerate(label_lst)}, )
    return config


def load_albert_tokenizer():
    tokenizer = AlbertTokenizer.from_pretrained("albert-large-v2")
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, e1_mask, e2_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SemEvalProcessor(object):
    """Processor for the Semeval data set """

    def __init__(self, args):
        self.args = args
        self.relation_labels = get_label(args)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = self.relation_labels.index(line[0])
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file
        return self._create_examples(self._read_tsv(os.path.join(self.args.data_dir, file_to_read)), mode)


processors = {"semeval": SemEvalProcessor}


def data_generator(
    examples,
    max_seq_len,
    tokenizer,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    add_sep_token=False,
    mask_padding_with_zero=True,
    batch_size=16
):
    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_label_ids = []
    all_e1_mask = []
    all_e2_mask = []

    while True:
        num = 1
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)

            e11_p = tokens_a.index("<e1>")  # the start position of entity1
            e12_p = tokens_a.index("</e1>")  # the end position of entity1
            e21_p = tokens_a.index("<e2>")  # the start position of entity2
            e22_p = tokens_a.index("</e2>")  # the end position of entity2

            # Replace the token
            tokens_a[e11_p] = "$"
            tokens_a[e12_p] = "$"
            tokens_a[e21_p] = "#"
            tokens_a[e22_p] = "#"

            # Add 1 because of the [CLS] token
            e11_p += 1
            e12_p += 1
            e21_p += 1
            e22_p += 1

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            if add_sep_token:
                special_tokens_count = 2
            else:
                special_tokens_count = 1
            if len(tokens_a) > max_seq_len - special_tokens_count:
                tokens_a = tokens_a[: (max_seq_len - special_tokens_count)]

            tokens = tokens_a
            if add_sep_token:
                tokens += [sep_token]

            token_type_ids = [sequence_a_segment_id] * len(tokens)

            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            # e1 mask, e2 mask
            e1_mask = [0] * len(attention_mask)
            e2_mask = [0] * len(attention_mask)

            for i in range(e11_p, e12_p + 1):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p + 1):
                e2_mask[i] = 1

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len
            )
            assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
                len(token_type_ids), max_seq_len
            )

            label_id = int(example.label)
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)
            all_label_ids.append(label_id)
            all_e1_mask.append(e1_mask)
            all_e2_mask.append(e2_mask)

            if num % batch_size == 0 or num == len(examples):
                yield [np.array(all_input_ids), np.array(all_attention_mask), np.array(all_token_type_ids), np.array(all_e1_mask), np.array(all_e2_mask)], np.array(all_label_ids)
                all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, all_e1_mask, all_e2_mask = [], [], [], [], [], []
            num += 1


def convert_examples_to_list(examples,
                             max_seq_len,
                             tokenizer,
                             cls_token="[CLS]",
                             cls_token_segment_id=0,
                             sep_token="[SEP]",
                             pad_token=0,
                             pad_token_segment_id=0,
                             sequence_a_segment_id=0,
                             add_sep_token=False,
                             mask_padding_with_zero=True):
    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_label_ids = []
    all_e1_mask = []
    all_e2_mask = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        e11_p = tokens_a.index("<e1>")  # the start position of entity1
        e12_p = tokens_a.index("</e1>")  # the end position of entity1
        e21_p = tokens_a.index("<e2>")  # the start position of entity2
        e22_p = tokens_a.index("</e2>")  # the end position of entity2

        # Replace the token
        tokens_a[e11_p] = "$"
        tokens_a[e12_p] = "$"
        tokens_a[e21_p] = "#"
        tokens_a[e22_p] = "#"

        # Add 1 because of the [CLS] token
        e11_p += 1
        e12_p += 1
        e21_p += 1
        e22_p += 1

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        if add_sep_token:
            special_tokens_count = 2
        else:
            special_tokens_count = 1
        if len(tokens_a) > max_seq_len - special_tokens_count:
            tokens_a = tokens_a[: (max_seq_len - special_tokens_count)]

        tokens = tokens_a
        if add_sep_token:
            tokens += [sep_token]

        token_type_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # e1 mask, e2 mask
        e1_mask = [0] * len(attention_mask)
        e2_mask = [0] * len(attention_mask)

        for i in range(e11_p, e12_p + 1):
            e1_mask[i] = 1
        for i in range(e21_p, e22_p + 1):
            e2_mask[i] = 1

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )

        label_id = int(example.label)
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_label_ids.append(label_id)
        all_e1_mask.append(e1_mask)
        all_e2_mask.append(e2_mask)
    return all_input_ids, all_attention_mask, all_token_type_ids, all_e1_mask, all_e2_mask, all_label_ids


class DataGenerator:
    def __init__(self, inputs, batch_size=16, is_test=False):
        self.inputs = inputs
        self.batch_size = batch_size
        self.steps = len(self.inputs[0]) // self.batch_size
        self.is_test = is_test
        if len(self.inputs[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.inputs[0])))
            if not self.is_test:
                np.random.shuffle(idxs)
            input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, label_ids = [], [], [], [], [], []
            for i in idxs:
                input_ids.append(self.inputs[0][i])
                attention_mask.append(self.inputs[1][i])
                token_type_ids.append(self.inputs[2][i])
                e1_mask.append(self.inputs[3][i])
                e2_mask.append(self.inputs[4][i])
                label_ids.append(self.inputs[5][i])
                if len(input_ids) == self.batch_size or i == idxs[-1]:
                    input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, label_ids = \
                        np.array(input_ids), np.array(attention_mask), np.array(token_type_ids), np.array(e1_mask), np.array(e2_mask), np.array(label_ids)
                    yield [input_ids, attention_mask, token_type_ids, e1_mask, e2_mask], label_ids
                    input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, label_ids = [], [], [], [], [], []


if __name__ == '__main__':
    import tensorflow as tf


    def count(stop):
        i = 0
        while i < stop:
            yield i
            i += 1


    ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes=(), )
    for x in ds_counter:
        print(x)
