"""_summary_

Returns:
    _type_: _description_
"""

from torch import nn
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
        self.final_layer = nn.Linear(hidden_size, target_vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):

        x = self.embed(x)
        out, hidden = self.rnn_unit(x, hidden)

        out = self.final_layer(out)

        out = self.softmax(out)
        return out, hidden
