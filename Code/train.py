"""_summary_

Returns:
    _type_: _description_
"""
import torch
from constants import TENSORBOARD_LOG_DIR
from data_loader import LoadAndData, TextData, get_data_generators
from loguru import logger
from model import Decoder, Encoder
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Train:
    """_summary_

    Returns:
        _type_: _description_
    """
    @property
    def data_loader(self):
        return self._data_loader

    @data_loader.setter
    def data_loader(self, loader):
        assert isinstance(
            loader, DataLoader
        ), f"Expected the data loader of type Dataset but got {type(loader)}"
        self._data_loader = loader

    @data_loader.getter
    def data_loader(self):
        return self._data_loader

    @property
    def tens_board(self):
        return self._tens_board

    @tens_board.setter
    def tens_board(self, enable):
        if enable:

            # Processing model name as model directory to store tensorboard logs
            model_dir = str(self).replace(" ", "_")
            tensorboard_dir = f"{self.log_dir}/{model_dir}"
            logger.log(self.log_level, f"tensorboard dir : {tensorboard_dir}")

            # Creating tensorboard writer object
            self._tens_board = self._tens_board_train = SummaryWriter(tensorboard_dir, comment='train')
            self._tens_board_validation = SummaryWriter(tensorboard_dir, comment='validation')
            self._tens_board_test = SummaryWriter(tensorboard_dir, comment='test')

            # Adding model graph to the tensorboard writer
            sample_img = iter(training_generator).next()[0]
            self._tens_board_train.add_graph(self.model, sample_img)
            logger.debug("Added graph...")

    @tens_board.getter
    def tens_board(self):
        return self._tens_board

    def __repr__(self) -> str:
        return self.name


class Trainer(Train):

    def __init__(self,
                 loss_function,
                 optimizer,
                 tensorboard=False,
                 log_dir=TENSORBOARD_LOG_DIR,
                 model_name='image to text',
                 log_level=5,
                 train_data=[],
                 validation_data=[],
                 test_data=[],
                 encoder=None,
                 decoder=None,
                 source_vocab_list=[],
                 destination_vocab_list=[]
                 ):

        self.name = model_name
        self.log_level = log_level
        super(Trainer, self).__init__()
        
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.loss_fun = loss_function
        self.opt = optimizer
        self.log_dir = log_dir
        #self.tens_board = tensorboard
        self.encoder = encoder
        self.decoder = decoder
        self.optimizers = optimizer
        self.train_loss = 0
        self.validation_loss = 0
        self.test_loss = 0
        self.epoch = 0
        self.encoding_model = Encoder(len(source_vocab_list), 5)

        self.decoding_model = Decoder(len(destination_vocab_list), 5, len(destination_vocab_list))

        self.tens_board_train = SummaryWriter(log_dir, comment='train')
        self.tens_board_validation = SummaryWriter(log_dir, comment='validation')
        self.tens_board_test = SummaryWriter(log_dir, comment='test')


    def __zero_grad__(self,):
        for optm in self.optimizers:
          optm.zero_grad()

    def __step_grad__(self,):
        for optm in self.optimizers:
          optm.step()

    def __iter_each_batch__(self, data_point, iter_for='train'):

        x = data_point[0]
        y = data_point[1]

        enc_out = torch.zeros(LoadAndData.MAX_SENT_LEN, 5)
        for e_w in range(1, x.shape[1]):
            # Iterating each word and tringing the encoder
            en_inp = x[:, e_w]
            encoding_out, encoding_hidden = self.encoding_model(en_inp)
            enc_out[e_w] = encoding_out[0,0]

        d_inp = y[:, 0]
        for d_w in range(1, y.shape[1]):
            d_out, d_hid = decoding_model(d_inp, encoding_hidden)
            d_inp = y[:, d_w]
            batch_loss = self.loss_fun(d_out, y[:, d_w])

        if iter_for == 'train':
            # Backpropagation
            batch_loss.backward()
        
        return round(batch_loss.item() / y.shape[1], 30)
    
    def __train__(self,):

        data_size = len(self.train_data)

        self.__zero_grad__()
        for batch_id, data_point in enumerate(self.train_data):

            batch_loss = self.__iter_each_batch__(data_point)
            
            self.epoch_loop.set_postfix_str(f"[Batch - {batch_id}] Loss : {batch_loss}")

        self.__step_grad__()

        self.__validate__()
    def __validate__(self,):
        logger.info(f"validating the model")
        data_size = len(self.validation_data.dataset)
        with torch.no_grad():
            for batch_id, data_point in enumerate(self.validation_data):

                batch_loss = self.__iter_each_batch__(data_point, iter_for='validate')
                self.epoch_loop.set_postfix_str(f"Validate Loss : {batch_loss}")        
        self.__test__()
    def __test__(self, ):
        data_size = len(self.test_data.dataset)
        with torch.no_grad():
            for batch_id, data_point in enumerate(self.test_data):

              batch_loss = self.__iter_each_batch__(data_point, iter_for='test')
              self.epoch_loop.set_postfix_str(f"Test Loss : {batch_loss}")               
    def train(self, epochs=10):
        """_summary_

        Args:
            epochs (int, optional): _description_. Defaults to 10.
        """
        self.epoch_loop = tqdm(range(epochs))
        for epoch in self.epoch_loop:
            self.train_loss = 0
            self.validation_loss = 0
            self.test_loss = 0
            self.epoch = epoch
            self.epoch_loop.set_description(f"Epoch : {epoch + 1}")
            
            self.__train__()


if __name__ == "__main__":
    
    training_generator, test_data_generator, validate_data_generator = get_data_generators()
    encoding_model = Encoder(len(TextData.source_vocab), 5)
    decoding_model = Decoder(len(TextData.destination_vocab), 5, len(TextData.destination_vocab))

    encoder_optimizer = torch.optim.SGD(encoding_model.parameters(), lr=0.002)
    decoder_optimizer = torch.optim.SGD(decoding_model.parameters(), lr=0.002)

    optimizers = [
    encoder_optimizer,
    decoder_optimizer
    ]

    loss_fun = nn.NLLLoss()
    trainer = Trainer(
        loss_fun,
        optimizers,
        tensorboard=True,
        log_dir=TENSORBOARD_LOG_DIR,
        model_name='translator_ec_dc',
        log_level=5,
        train_data=training_generator,
        validation_data=validate_data_generator,
        test_data=test_data_generator,
        encoder=encoding_model,
        decoder=decoding_model
        )

    trainer.train(1)
    
    