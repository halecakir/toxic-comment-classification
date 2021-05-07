import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ToxicCommentClassifier(nn.Module):
    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        output_size,
        vocab,
        dropout_rate=0.2,
    ):
        super(ToxicCommentClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.embeddings = nn.Embedding(
            num_embeddings=len(vocab.w2i),
            embedding_dim=embedding_size,
            padding_idx=vocab.pad_id,
        )
        self.encoder = nn.LSTM(
            input_size=embedding_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.1,
        )

        self.output_projection = nn.Linear(
            in_features=2 * self.hidden_size, out_features=output_size, bias=False
        )

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.__initParameters()

    def __initParameters(self):
        for name, param in self.named_parameters():
            if "bias" not in name and param.requires_grad:
                init.xavier_uniform_(param)

    def initalize_pretrained_embeddings(self, embeddings):
        weights_matrix = np.zeros(((len(self.vocab.w2i), self.embedding_size)))
        count = 0
        for word in self.vocab.w2i:
            if word in embeddings:
                count += 1
                weights_matrix[self.vocab.w2i[word]] = embeddings[word]
        self.embeddings.weight = nn.Parameter(torch.FloatTensor(weights_matrix))
        return count, len(self.vocab.w2i)

    def forward(self, inputs):
        inputs_lengths = self.vocab.to_input_lengths(inputs)

        # Inputs padded shape : (seq_len, batch_size)
        inputs_padded = self.vocab.to_input_tensor(inputs)

        # X tensor shape : (seq_len, batch_size, embedding_size)
        X = self.embeddings(inputs_padded)

        X = self.dropout(X)

        # Pack the paddings to reduce the number of computations
        packet_X = pack_padded_sequence(X, inputs_lengths)

        enc_hiddens_packet, (last_hidden, last_cell) = self.encoder(packet_X)
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens_packet)
        last_hidden = torch.cat((last_hidden[0], last_hidden[1]), 1)

        h2o = self.output_projection(last_hidden)
        pred = self.sigmoid(h2o)

        return pred
