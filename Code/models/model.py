"""_summary_

Returns:
    _type_: _description_
"""

from torch import nn
from torch.nn.functional import relu, tanh
from loguru import logger


class Encoder(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()

        self.embed = nn.Embedding(input_size, hidden_size,)

        self.rnn_unit = nn.RNN(hidden_size, hidden_size,
                               dropout=0.2, num_layers=2)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.embed(x)
        out, hidden = self.rnn_unit(x)

        return out, hidden


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, target_vocab_size):
        super(Decoder, self).__init__()

        self.embed = nn.Embedding(input_size, hidden_size,)

        self.rnn_unit = nn.RNN(hidden_size, hidden_size,
                               dropout=0.3, num_layers=2)
        self.final_layer = nn.Linear(hidden_size, target_vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):

        x = self.embed(x)
        out, hidden = self.rnn_unit(x, hidden)

        out = self.final_layer(out)

        out = self.softmax(out)
        return out, hidden


class BiEncoder(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, input_size, hidden_size):
        super(BiEncoder, self).__init__()

        self.embed = nn.Embedding(input_size, hidden_size,)

        self.rnn_unit = nn.RNN(hidden_size, hidden_size,
                               dropout=0.2, num_layers=2, bidirectional=True)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.embed(x)
        out, hidden = self.rnn_unit(x)

        return out, hidden


class BiDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, target_vocab_size):
        super(BiDecoder, self).__init__()

        self.embed = nn.Embedding(input_size, hidden_size,)

        self.rnn_unit = nn.RNN(hidden_size, hidden_size,
                            dropout=0.3, num_layers=2, bidirectional=True)
        self.rnn_unit2 = nn.RNN(hidden_size, hidden_size,
                            dropout=0.2, num_layers=2, bidirectional=True)
        
        self.final_layer = nn.Linear(hidden_size * 2, target_vocab_size)
        self.batch_normal = nn.BatchNorm1d(hidden_size * 2)
        self.drop_out = nn.Dropout(0.5)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):

        x = self.embed(x).view(1,1,-1)
        out, hidden = self.rnn_unit(x, hidden)

        out = self.drop_out(out)
        
        # out = self.batch_normal(out)
        out = self.final_layer(out)
        out = self.softmax(out)
        return out, hidden




class LSTMEncoder(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, input_size, hidden_size):
        super(LSTMEncoder, self).__init__()

        self.embed = nn.Embedding(input_size, hidden_size,)

        self.rnn_unit = nn.LSTM(hidden_size, hidden_size,
                                dropout=0.1, num_layers=2, batch_first=True)
        self.rnn_unit2 = nn.LSTM(hidden_size, hidden_size,
                                dropout=0.2, num_layers=2, batch_first=True)
        self.rnn_unit3 = nn.LSTM(hidden_size, hidden_size,
                                dropout=0.1, num_layers=2, batch_first=True)
        
        
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.embed(x)
        x = relu(x)
        out, (h_n, c_n) = self.rnn_unit(x)
        out = self.batch_norm(out)
        out, (h_n, c_n) = self.rnn_unit2(out)
        out = self.batch_norm2(out)
        out, (h_n, c_n) = self.rnn_unit3(out)
        
        return out, h_n, c_n


class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, target_vocab_size):
        super(LSTMDecoder, self).__init__()

        self.embed = nn.Embedding(input_size, hidden_size,)

        self.rnn_unit = nn.LSTM(hidden_size, hidden_size,
                                dropout=0.3, num_layers=2, batch_first=True)
        
        self.rnn_unit2 = nn.LSTM(hidden_size, hidden_size,
                                dropout=0.3, num_layers=2, batch_first=True)
        self.final_layer = nn.Linear(hidden_size, target_vocab_size)
        self.batch_normal = nn.BatchNorm1d(hidden_size)
        self.batch_normal2 = nn.BatchNorm1d(target_vocab_size)
        self.drop_out = nn.Dropout(0.5)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h_n, c_n):

        x = self.embed(x)
        x = relu(x)
        out, (h_n, c_n) = self.rnn_unit(x, (h_n, c_n))

        # out = self.drop_out(out)
        
        out = self.batch_normal(out)
        
        out, (h_n, c_n) = self.rnn_unit2(out, (h_n, c_n))
        out = relu(out)
        out = self.final_layer(out)
        out = self.batch_normal2(out)
        out = self.softmax(out)
        return out, h_n, c_n
