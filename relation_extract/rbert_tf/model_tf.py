from transformers import TFBertModel, TFBertPreTrainedModel, TFAlbertModel, TFAlbertPreTrainedModel
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Layer

"""
参考论文：
    Enriching Pre-trained Language Model with Entity Information for Relation Classification
"""


class FCLayer(Layer):
    def __init__(self, output_dim, dropout_rate=0.1, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = Dropout(dropout_rate)
        self.dense = Dense(output_dim)

    # def build(self, input_shape):
    #     self.weight = self.add_weight(shape=(input_shape[-1], self.output_dim), trainable=True)
    #     super(FCLayer, self).build(input_shape)

    def call(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = tf.nn.tanh(x)
        return self.dense(x)


class RBERT(TFBertPreTrainedModel):
    def __init__(self, config, args):
        super(RBERT, self).__init__(config)
        self.bert = TFBertModel(config=config)
        self.num_labels = config.num_labels
        self.cls_fc_layer = FCLayer(config.hidden_size, args.dropout_rate)
        self.entity_fc_layer = FCLayer(config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(config.hidden_size, args.dropout_rate, use_activation=False)
        self.dense = Dense(config.num_labels, activation="softmax")

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = tf.expand_dims(e_mask, axis=1)  # [b,1,j-i+1]
        length_tensor = tf.expand_dims(tf.reduce_sum(tf.cast((e_mask != 0), tf.float32), axis=1), 1)  # [batch_size,1]
        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = tf.squeeze(tf.matmul(tf.cast(e_mask_unsqueeze, tf.float32), hidden_output), axis=1)
        avg_vector = sum_vector / length_tensor  # broadcasting
        return avg_vector

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids, e1_mask, e2_mask = inputs
        outputs = self.bert([input_ids, attention_mask, token_type_ids])
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer(Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # # Concat ->fc_layer
        concat_h = tf.concat([pooled_output, e1_h, e2_h], axis=-1)
        logits = self.label_classifier(concat_h)
        logits = self.dense(logits)  # 原始的pytorch实现中没有这一层，我这里使用tf.keras实现不添加这一层训练效果很差，暂不知道原因
        return logits


class RALBERT(TFAlbertPreTrainedModel):
    def __init__(self, config, args):
        super(RALBERT, self).__init__(config)
        self.bert = TFAlbertModel(config=config)
        self.num_labels = config.num_labels
        self.cls_fc_layer = FCLayer(config.hidden_size, args.dropout_rate)
        self.entity_fc_layer = FCLayer(config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(config.hidden_size, args.dropout_rate, use_activation=False)
        self.dense = Dense(config.num_labels, activation="softmax")

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = tf.expand_dims(e_mask, axis=1)  # [b,1,j-i+1]
        length_tensor = tf.expand_dims(tf.reduce_sum(tf.cast((e_mask != 0), tf.float32), axis=1), 1)  # [batch_size,1]
        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = tf.squeeze(tf.matmul(tf.cast(e_mask_unsqueeze, tf.float32), hidden_output), axis=1)
        avg_vector = sum_vector / length_tensor  # broadcasting
        return avg_vector

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids, e1_mask, e2_mask = inputs
        outputs = self.bert([input_ids, attention_mask, token_type_ids])
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer(Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # # Concat ->fc_layer
        concat_h = tf.concat([pooled_output, e1_h, e2_h], axis=-1)
        logits = self.label_classifier(concat_h)
        logits = self.dense(logits)  # 原始的pytorch实现中没有这一层，我这里使用tf.keras实现不添加这一层训练效果很差，暂不知道原因
        return logits
