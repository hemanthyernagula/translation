"""_summary_

Returns:
    _type_: _description_
"""

from torch import nn
from torch.nn.functional import relu, tanh
from loguru import logger
from constants import DEFAULT_SCORING_FUNCTION, PAD_TOKEN_INDEX, BATCH_SIZE, NO_OF_LSTM_LAYERS
from typing import AnyStr, Dict
import torch

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

    def __init__(self, input_size, hidden_size, max_sent_len=8):
        super(LSTMEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.no_of_layers = NO_OF_LSTM_LAYERS
        self.embed = nn.Embedding(input_size, self.hidden_size,)

        self.rnn_unit = nn.LSTM(self.hidden_size, self.hidden_size,
                                dropout=0.1, num_layers=self.no_of_layers, batch_first=True)
        self.rnn_unit2 = nn.LSTM(self.hidden_size, self.hidden_size,
                                dropout=0.2, num_layers=self.no_of_layers, batch_first=True)
        self.rnn_unit3 = nn.LSTM(self.hidden_size, self.hidden_size,
                                dropout=0.1, num_layers=self.no_of_layers, batch_first=True)
        
        
        self.batch_norm = nn.BatchNorm1d(max_sent_len)
        self.batch_norm2 = nn.BatchNorm1d(max_sent_len)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, h_n, c_n):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # x shape --> [BATCH_SIZE, MAX_LEN_OF_SENT]
        x = self.embed(x) # [BATCH_SIZE, MAX_LEN_OF_SENT, HIDDEN_SIZE]
        out, (h_n, c_n) = self.rnn_unit(x, (h_n, c_n))
        
        # out --> # [BATCH_SIZE, MAX_LEN_OF_SENT, HIDDEN_SIZE]
        # h_n --> # [1* NO_OF_LSTM_LAYERS, BATCH_SIZE, HIDDEN_SIZE]
        # c_n --> # [1* NO_OF_LSTM_LAYERS, BATCH_SIZE, HIDDEN_SIZE]
        
        out = self.batch_norm(out)
        
        # out --> [BATCH_SIZE, MAX_LEN_OF_SENT, HIDDEN_SIZE]
        out = relu(out)
        out, (h_n, c_n) = self.rnn_unit2(out, (h_n, c_n))
        out = self.batch_norm2(out)
        out, (h_n, c_n) = self.rnn_unit3(out, (h_n, c_n))
        
        return out, h_n, c_n

    def init_hidden_state(self,):
        return torch.ones(self.no_of_layers, BATCH_SIZE, self.hidden_size) * PAD_TOKEN_INDEX
class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, target_vocab_size):
        super(LSTMDecoder, self).__init__()

        self.embed = nn.Embedding(input_size, hidden_size,)
        self.hidden_size = hidden_size
        self.no_of_layers = NO_OF_LSTM_LAYERS

        self.rnn_unit = nn.LSTM(hidden_size, hidden_size,
                                dropout=0.2, num_layers=2, batch_first=True)
        
        self.rnn_unit2 = nn.LSTM(hidden_size, hidden_size,
                                dropout=0.3, num_layers=2, batch_first=True)
        self.final_layer = nn.Linear(hidden_size, target_vocab_size)
        self.batch_normal = nn.BatchNorm1d(hidden_size)
        self.batch_normal2 = nn.BatchNorm1d(target_vocab_size)
        self.drop_out = nn.Dropout(0.5)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def init_hidden_state(self,):
        return torch.ones(self.no_of_layers, BATCH_SIZE,self.hidden_size ) * PAD_TOKEN_INDEX

    def forward(self, x, h_n, c_n, encoder_outputs):

        x = self.embed(x)
        
        x = relu(x)
        out, (h_n, c_n) = self.rnn_unit(x, (h_n, c_n))

        # out = self.drop_out(out)
        
        out = self.batch_normal(out)
        
        out, (h_n, c_n) = self.rnn_unit2(out, (h_n, c_n))
        out = relu(out)       
        logger.info(f"out size before final layer : {out.shape}")
        out = self.final_layer(out)
        logger.info(f"out size after final layer : {out.shape}")
        out = self.batch_normal2(out)
        logger.info(f"out size after batch_norm layer : {out.shape}")
        out = self.softmax(out)
        logger.info(f"out size after softmax layer : {out.shape}")
        return out, h_n, c_n


class LSTMAttentionDecoder(LSTMDecoder):
        
    def __init__(self, input_size, hidden_size, target_vocab_size, attention):
        super(LSTMDecoder, self).__init__()

        self.attention = attention(hidden_size)
        self.embed = nn.Embedding(input_size, hidden_size,)
        self.hidden_size = hidden_size
        self.no_of_layers = NO_OF_LSTM_LAYERS

        self.rnn_unit = nn.LSTM(hidden_size*2, hidden_size,
                                dropout=0.2, num_layers=1, batch_first=True)
        
        self.rnn_unit2 = nn.LSTM(hidden_size, hidden_size,
                                dropout=0.3, num_layers=2, batch_first=True)
        
        self.final_layer = nn.Linear(hidden_size, target_vocab_size)
        self.batch_normal = nn.BatchNorm1d(hidden_size)
        self.batch_normal2 = nn.BatchNorm1d(target_vocab_size)
        self.drop_out = nn.Dropout(0.5)
        self.softmax = nn.LogSoftmax(dim=0)
        
    def init_hidden_state(self,):
        return torch.ones(self.no_of_layers, BATCH_SIZE,self.hidden_size ) * PAD_TOKEN_INDEX

    def forward(self, x, h_n, c_n, encoder_outputs):

        # x --> [BATCH_SIZE, 1]
        # h_n --> # [1* NO_OF_LSTM_LAYERS, BATCH_SIZE, HIDDEN_SIZE]
        # c_n --> # [1* NO_OF_LSTM_LAYERS, BATCH_SIZE, HIDDEN_SIZE]
        # encoder_outputs --> [BATCH_SIZE, MAX_LEN_OF_SENT, HIDDEN_SIZE]
        embed_x = self.embed(x) # [BATCH_SIZE, HIDDEN_SIZE]
        
        
        context_vector, att_wts = self.attention(encoder_outputs, decoder_hidden=h_n)
        
        # allignment_scores [BATCH_SIZE, MAX_LEN_OF_SENT]
        # context_vector --> [BATCH_SIZE, 1, HIDDEN_SIZE]
        
        decoder_input = torch.cat((embed_x, context_vector[:, 0]), 1).unsqueeze(1)
        
        # decoder_input(after unsqueeze 0) --> [1, BATCH_SIZE, HIDDEN_SIZE*2]
        
        out, (h_n, c_n) = self.rnn_unit(decoder_input, (h_n, c_n))
        
        # logger.info(f"out size before final layer : {out.shape}")

        # out --> [BATCH_SIZE, 1, HIDDEN_SIZE]

        # out, (h_n, c_n) = self.rnn_unit2(out, (h_n, c_n))
        out = relu(out)
        out = self.final_layer(out).squeeze() # --> [BATCH_SIZE, target_vocab_size]
        # logger.info(f"out size after final layer : {out.shape}")
        # out = self.batch_normal2(out)
        out = self.softmax(out)
        # logger.info(f"out size after softmax layer : {out.shape}")
        return out, h_n, c_n, att_wts


class Attention(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, hidden) -> None:
        super(Attention, self).__init__()
        
        self.hidden = hidden
        self.decoder_trainable = nn.Linear(hidden, hidden, bias=False)
        self.encoder_trainable = nn.Linear(hidden, hidden, bias=False)
        self.alignment_trainable = nn.Linear(hidden, 1, bias=False)
        self.cos = nn.CosineSimilarity(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_outputs, decoder_hidden, scoring_type=DEFAULT_SCORING_FUNCTION):
        
        
        # temp = nn.Parameter(torch.FloatTensor(1, self.hidden))
        # combined_activations.bmm(temp.unsqueeze(2)).shape
        
        decoder_hidden = decoder_hidden.view(BATCH_SIZE, -1, self.hidden)
        decoder_trained = self.decoder_trainable(decoder_hidden)
        encoder_trained = self.encoder_trainable(encoder_outputs)
        
        # decoder_trained --> [BATCH_SIZE, 1* NO_OF_LSTM_LAYERS, HIDDEN_SIZE]
        # encoder_trained --> [BATCH_SIZE, MAX_LEN_OF_SENT, HIDDEN_SIZE]
        
        combined_activations = torch.tanh(
            decoder_trained + encoder_trained
        )
        
        # combined_actions --> [BATCH_SIZE, MAX_LEN_OF_SENT, HIDDEN_SIZE]
        
        allignments = self.alignment_trainable(combined_activations) # [BATCH_SIZE, MAX_LEN_OF_SENT, 1]
        allignment_scores = self.softmax(allignments.view(1, -1)).view(BATCH_SIZE, -1)
        
        # After unsqeeze allignment_scores [1, BATCH_SIZE, MAX_LEN_OF_SENT]
        context_vector = torch.bmm(allignment_scores.unsqueeze(1), encoder_outputs)
        
        # context_vector --> [BATCH_SIZE, 1, HIDDEN_SIZE]
        return context_vector, allignment_scores
    
    def score_fun(self, fun_type: AnyStr=DEFAULT_SCORING_FUNCTION):
        """_summary_

        Args:
            fun_type (AnyStr, optional): _description_. Defaults to DEFAULT_SCORING_FUNCTION.

        Returns:
            _type_: _description_
        """
        if fun_type == DEFAULT_SCORING_FUNCTION:
            return self.cos