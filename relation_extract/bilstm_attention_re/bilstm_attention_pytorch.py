import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

is_cuda_ava = True if torch.cuda.is_available() else False


class BiLSTMAttention(nn.Module):
    def __init__(self, config, embedding_pre):
        super(BiLSTMAttention, self).__init__()
        self.batch = config["BATCH"]
        self.embedding_size = config["EMBEDDING_SIZE"]
        self.embedding_dim = config["EMBEDDING_DIM"]
        self.hidden_dim = config["HIDDEN_DIM"]
        self.tag_size = config["TAG_SIZE"]
        self.pos_size = config["POS_SIZE"]
        self.pos_dim = config["POS_DIM"]
        self.pretrained = config["pretrained"]
        if self.pretrained:
            self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre), freeze=False)
        else:
            self.word_embeds = nn.Embedding(self.embedding_size, self.embedding_dim)

        self.pos1_embeds = nn.Embedding(self.pos_size, self.pos_dim)
        self.pos2_embeds = nn.Embedding(self.pos_size, self.pos_dim)
        self.relation_embeds = nn.Embedding(self.tag_size, self.hidden_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim + self.pos_dim * 2, hidden_size=self.hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.dropout_emb = nn.Dropout(p=0.5)
        self.dropout_lstm = nn.Dropout(p=0.5)
        self.dropout_att = nn.Dropout(p=0.5)
        self.hidden = self.init_hidden()
        self.att_weight = nn.Parameter(torch.randn(self.batch, 1, self.hidden_dim))  # 需要训练
        self.relation_bias = nn.Parameter(torch.randn(self.batch, self.tag_size, 1))

    def init_hidden(self):
        return torch.randn(2, self.batch, self.hidden_dim // 2)  # 初始状态，不用训练

    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch, self.hidden_dim // 2),  # 2表示双向 shape=(2,batch_size,hidden_dim/2)
                torch.randn(2, self.batch, self.hidden_dim // 2))  # lstm的初始状态，不用训练

    def attention(self, H):
        """
        计算注意力
        :param H:
        :return:
        """
        M = F.tanh(H)
        a = F.softmax(torch.bmm(self.att_weight, M), 2)  # shape=[batch_size,1,seq_len]
        a = torch.transpose(a, 1, 2)  # shape=[batch_size,seq_len,1]
        return torch.bmm(H, a)

    def forward(self, sentence, pos1, pos2):
        self.hidden = self.init_hidden_lstm()  # shape=(2,batch_size,hidden_dim/2)
        if is_cuda_ava:
            self.hidden = self.hidden[0].cuda(), self.hidden[1].cuda()
            sentence = sentence.to(torch.long).cuda()
            pos1 = pos1.to(torch.long).cuda()
            pos2 = pos2.to(torch.long).cuda()
            print("hidden shape:", self.hidden[0].shape)
            embeds = torch.cat((self.word_embeds(sentence),
                                self.pos1_embeds(pos2), self.pos2_embeds(pos2)), 2)  # shape=[batch_size,seq_len,embedding_dim+pos_dim*2]
            embeds = torch.transpose(embeds, 0, 1)  # shape=[seq_len,batch_size,embedding_dim+pos_dim*2]
            print("embeds shape:", embeds.shape)
            lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # lstm_out shape=[seq_len,batch_size,hidden_dim]
            lstm_out = torch.transpose(lstm_out, 0, 1)  # shape=[batch_size,seq_len,hidden_dim]
            print("lstm_out shape:", lstm_out.shape)
            lstm_out = torch.transpose(lstm_out, 1, 2)  # shape=[batch_size,hidden_dim,seq_len]
            print("lstm_out shape:", lstm_out.shape)
            lstm_out = self.dropout_lstm(lstm_out)  # shape=[batch_size,hidden_dim,seq_len]
            att_out = F.tanh(self.attention(lstm_out))  # shape=[batch_size,hidden_dim,1]
            print("att_out shape:", att_out.shape)

            relation = torch.tensor([i for i in range(self.tag_size)], dtype=torch.long).repeat(self.batch, 1)
            print("relation shape:", relation.shape)
            if is_cuda_ava:
                relation = relation.cuda()
            relation = self.relation_embeds(relation)
            print("relation embedding:", relation.shape)
            res = torch.add(torch.bmm(relation, att_out), self.relation_bias)  # shape=[batch_size,tag_size,1]
            res = F.softmax(res, 1)
        return res.view(self.batch, -1)


if __name__ == '__main__':
    EMBEDDING_SIZE = 100
    EMBEDDING_DIM = 100

    POS_SIZE = 82  # 不同数据集这里可能会报错。
    POS_DIM = 100

    HIDDEN_DIM = 200

    TAG_SIZE = 12

    BATCH = 128
    EPOCHS = 100

    config = {}
    config['EMBEDDING_SIZE'] = EMBEDDING_SIZE
    config['EMBEDDING_DIM'] = EMBEDDING_DIM
    config['POS_SIZE'] = POS_SIZE
    config['POS_DIM'] = POS_DIM
    config['HIDDEN_DIM'] = HIDDEN_DIM
    config['TAG_SIZE'] = TAG_SIZE
    config['BATCH'] = BATCH
    config["pretrained"] = False

    learning_rate = 0.0005
    model = BiLSTMAttention(config, "")
    model.to("cuda")
    # for x in model.parameters():
    #     print(x.shape)

    sentence = torch.ones(BATCH, 50).to(torch.long)
    pos1 = torch.ones(BATCH, 50).to(torch.long)
    pos2 = torch.ones(BATCH, 50).to(torch.long)

    model(sentence, pos1, pos2)
    # for x in model.named_parameters():
    #     print(x[0], x[1].shape)
