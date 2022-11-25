"""_summary_

Returns:
    _type_: _description_
"""
import json
import sys
from typing import AnyStr, Dict
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.text.bleu import BLEUScore
from tqdm import tqdm

from constants import (CONFIGS_PATH, EOS_TOKEN, EOS_TOKEN_INDEX, EPOCHS,
                       PAD_TOKEN, PAD_TOKEN_INDEX, SOS_TOKEN, SOS_TOKEN_INDEX,
                       TENSORBOARD_LOG_DIR, UKN_TOKEN, UKN_TOKEN_INDEX, HIDDEN_SIZE)
from data_loader import LoadAndData, TextData, get_data_generators
from models.model import (BiDecoder, BiEncoder, Decoder, Encoder, LSTMDecoder,
                          LSTMEncoder)
from mylogging import logger
from utils.load_configs import Configs

# logger.add(sys.stdout, format="{time} - {level} - {message}", filter="sub.module")
# logger.add("file_{time}.log", level="ERROR", rotation="100 MB")


Decoder = BiDecoder
Encoder = BiEncoder

Decoder = LSTMDecoder
Encoder = LSTMEncoder



class Train:

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
            
            logger.info(f"tensorboard dir : {self.model_path}")

            # Creating tensorboard writer object
            self._tens_board = self._tens_board_train = SummaryWriter(self.model_path, comment='train')
            self._tens_board_validation = SummaryWriter(self.model_path, comment='validation')
            self._tens_board_test = SummaryWriter(self.model_path, comment='test')

            # Adding model graph to the tensorboard writer
            inp = torch.ones(64, dtype=torch.long)
            hidden = torch.ones(2,256)
            
            
            # self._tens_board_train.add_graph(self.encoding_model, inp)
            self._tens_board_train.add_graph(self.decoding_model, (inp,hidden,hidden))
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
                 model_name='translator',
                 log_level=5,
                 train_data=None,
                 validation_data=None,
                 test_data=None,
                 encoder:  object = None,
                 decoder:  object = None,
                 source_vocab_list = [],
                 destination_vocab_list = []
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
        
        self.optimizers = optimizer
        

        self.encoding_model = encoder
        self.decoding_model = decoder
        self.train_loss = 0
        self.validation_loss = 0
        self.test_loss = 0
        
        self.epoch = 0

        self.model_path = log_dir + "/" + model_name        
        self.tens_board = tensorboard

        self.tens_board_train = SummaryWriter(self.model_path, comment='train')
        self.tens_board_validation = SummaryWriter(self.model_path, comment='validation')
        self.tens_board_test = SummaryWriter(self.model_path, comment='test')
        self.metric = BLEUScore()


    def __zero_grad__(self,):
        for optm in self.optimizers:
          optm.zero_grad()

    def __step_grad__(self,):
        for optm in self.optimizers:
          optm.step()

    def __print_encoder_inp__(self, inp):
        temp = []
        for i in inp:
            temp.append(source_vocab_list[i])

        logger.info(f"encoder inp : {' '.join(temp)}")
    def __iter_each_batch__(self, data_point, iter_for='train'):

        x = data_point[0]
        y = data_point[1] 

        enc_out = torch.zeros(LoadAndData.MAX_SENT_LEN, 5)
        for e_w in range(1, x.shape[1]):
            # Iterating each word and tringing the encoder
            en_inp = x[:, e_w]
            encoding_out, encoding_hidden = self.encoding_model(en_inp)
            enc_out[e_w] = encoding_out[0,0]
        
        d_hid = encoding_hidden
        d_inp = y[:, 0]
        for d_w in range(1, y.shape[1]):
            d_out, d_hid = decoding_model(d_inp, d_hid)
            d_inp = y[:, d_w]
            batch_loss = self.loss_fun(d_out, y[:, d_w])
            
            topv, topi = d_out.data.topk(1)
            # logger.info(f"topi : {topi}")
        if iter_for == 'train':
            # Backpropagation
            batch_loss.backward()

        self._batch_loss = batch_loss
        return round(batch_loss.item() / y.shape[1], 30)

    def __lstm_iter_each_batch__(self, data_point, iter_for='train'):

        x = data_point[0]
        y = data_point[1]
        encoding_h_n = None
        encoding_c_n = None

        batch_decoder_outputs = torch.ones_like(y) * PAD_TOKEN_INDEX
        batch_decoder_outputs[:, 0] = SOS_TOKEN_INDEX
        enc_out = torch.ones(LoadAndData.MAX_SENT_LEN, 5) * PAD_TOKEN_INDEX
        for e_w in range(0, x.shape[1]):
            # Iterating each word and tringing the encoder
            en_inp = x[:, e_w]
            encoding_out, encoding_h_n, encoding_c_n = self.encoding_model(en_inp)
            enc_out[e_w] = encoding_out[0,0]
            #logger.critical(f"for encoder input {en_inp} -->  output {enc_out}")

        d_h_n = encoding_h_n
        d_c_n = encoding_c_n
        d_inp = y[:, 0]
        batch_loss = 0
        for d_w in range(1, y.shape[1]):
            d_out, d_h_n, d_c_n = self.decoding_model(d_inp, d_h_n, d_c_n)
            batch_loss += self.loss_fun(d_out, y[:, d_w])
            #logger.critical(f"for decoder input : {d_out}")
            d_inp = y[:, d_w]
            topv, topi = d_out.data.topk(1)
            
            np.savetxt(f"debugging/{self.name}/pred_{d_w}_loss.txt", topi.squeeze())
            np.savetxt(f"debugging/{self.name}/actual_{d_w}_loss.txt", y[:, d_w].detach().numpy())
            batch_decoder_outputs[:, d_w] = topi.squeeze()
            
            #logger.critical(f"the decoder output is : {topi}")
        if iter_for == 'train':
            # Backpropagation
            batch_loss.backward()
            

        score = self.calculate_score(y, batch_decoder_outputs)
        
        self.add_text_to_tensorboard(x, batch_decoder_outputs)
        self._batch_loss = batch_loss
        return round(batch_loss.item() / y.shape[1], 30), score

    def calculate_score(self, actual, predicted):
        score = 0
        for pair in zip(actual, predicted):
            actual_vect = list(pair[0].numpy())
            pred = list(pair[1].numpy())
            
            actual_vect_str = [[" ".join([str(i) for i in actual_vect])]]
            pred_str = [" ".join([str(i) for i in pred])]
    
            with open(f"debugging/{self.name}/debugging.txt","a") as f:
                f.write(f"{actual_vect_str} --> {pred_str}\n")
                    
            score += self.metric(pred_str, actual_vect_str)
            
        return score / len(actual)

    def add_text_to_tensorboard(self, source, destination):
        for _, pair in enumerate(zip(source, destination)):
            source_vect = list(pair[0].numpy())
            destination_vect = list(pair[1].numpy())
            
            actual_str = ' '.join([TextData.source_vocab[i] for i in source_vect])
            destination_str = ' '.join([TextData.destination_vocab[i] for i in destination_vect])
            
            self.tens_board_train.add_text(f"Text_{self.epoch}", f"{actual_str} --> {destination_str}", _)

            
            
    def __batch_trainer__(self, ):
        return self.__lstm_iter_each_batch__
    
    def __project_weights__(self, ):
        """Plots the histogram for encoder and decoder for better understanding of weight updates
        """
        
            
        # Ref:- https://stackoverflow.com/questions/54817864/pytorch-0-4-1-lstm-object-has-no-attribute-weight-ih-l
        
        self.tens_board_train.add_histogram(
                'encoder_embedings', self.encoding_model.embed.weight, global_step=self.epoch
            )
        
        # for layers in self.encoding_model.rnn_unit._all_weights:
        #     if isinstance(layers, list):
        #         for each_layer in layers:
        #             self.tens_board_train.add_histogram(
        #                 "encoder_" + str(each_layer), self.encoding_model.rnn_unit._parameters[each_layer],
        #                 )
                    
                    
        self.tens_board_train.add_histogram('decoder_embedings', self.decoding_model.embed.weight, global_step=self.epoch)
        # for layers in self.decoding_model.rnn_unit._all_weights:
        #     if isinstance(layers, list):
        #         for each_layer in layers:
        #             self.tens_board_train.add_histogram(
        #                 "decoder_"+str(each_layer), self.decoding_model.rnn_unit._parameters[each_layer],
        #                 )
        
        self.tens_board_train.add_histogram("decoder_final_layer_weight", self.decoding_model.final_layer.weight, global_step=self.epoch)
        self.tens_board_train.add_histogram("decoder_final_layer_bias", self.decoding_model.final_layer.bias, global_step=self.epoch)
        
            
    
    def __train__(self,):

        data_size = len(self.train_data)

        self.__zero_grad__()
        for batch_id, data_point in enumerate(self.train_data):

            batch_loss, score = self.__batch_trainer__()(data_point)
            self.all_loss.update({"Train Loss" : batch_loss})
            self.all_score.update({
                "Train Score": score
            })

            self.progress_dict.update({
                **self.all_loss,
                "Batch":batch_id
            })
            self.epoch_loop.set_postfix(self.progress_dict)
            self.__project_weights__()

        self.__step_grad__()

        self.__validate__()
        
        # logger.info(f"tensorboard logging")
        self.tens_board_train.add_scalars(
            'Loss',
            self.all_loss,
            global_step = self.epoch
        )
        
        self.tens_board_train.add_scalars(
            'BLEU',
            self.all_score,
            global_step = self.epoch
        )
        
        
        
    def __validate__(self,):
        data_size = len(self.validation_data.dataset)
        with torch.no_grad():
            for batch_id, data_point in enumerate(self.validation_data):

                batch_loss, score = self.__batch_trainer__()(data_point, iter_for='validate')
                self.all_loss.update({"Validate Loss" : batch_loss})
                self.all_score.update({
                    "Validate Score": score
                })
                self.progress_dict.update(self.all_loss)
                self.epoch_loop.set_postfix(self.progress_dict)     

    def __test__(self, ):
        data_size = len(self.test_data.dataset)
        with torch.no_grad():
            for batch_id, data_point in enumerate(self.test_data):

                batch_loss, score = self.__batch_trainer__()(data_point, iter_for='test')
                self.all_loss.update({"Test Loss": batch_loss})
                self.all_score.update({
                    "Test Score": score
                })
                self.progress_dict.update(self.all_loss)
                self.epoch_loop.set_postfix(self.progress_dict)       
    def train(self, epochs=10):
        self.epoch_loop = tqdm(range(epochs))
        self.all_loss = {
                    "Train Loss" : 0,
                    "Validate Loss": 0,
                    "Test Loss": 0
        }
        
        self.all_score = {
            "Train Score": 0,
            "Validate Score": 0,
            "Test Score": 0
        }
        self.progress_dict = {
                                "Batch":0,
                                **self.all_loss,
                            }
        
        self.tens_board_train.add_embedding(self.encoding_model.embed.weight,
                        metadata  = np.arange(self.encoding_model.embed.weight.shape[0]),
                        tag = f'encoder embeding')
        
        
        self.tens_board_train.add_embedding(self.decoding_model.embed.weight,
                        metadata  = np.arange(self.decoding_model.embed.weight.shape[0]),
                        tag = f'decoding embeding')
        
        create_dir(f"debugging/{self.name}")
        for epoch in self.epoch_loop:
            model_path = f"./model_files/{self.name}/{epoch}/"
            os.makedirs(model_path, exist_ok=True)
            self.train_loss = 0
            self.validation_loss = 0
            self.test_loss = 0
            self.epoch = epoch
            self.epoch_loop.set_description(f"Epoch : {epoch + 1}")
            self.epoch_loop.set_postfix(self.progress_dict)
            self.__train__()
            self.__test__()
            
            decoder_path = model_path + "decoder.pt"
            encoder_path = model_path + "encoder.pt"

            torch.save(self.encoding_model.state_dict(), encoder_path)
            torch.save(self.decoding_model.state_dict(), decoder_path)
            
# %%
def create_dir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            logger.debug("the directory already exists")
        
    
def make_training_ready(model_name):
    
    training_generator, test_data_generator, validate_data_generator = get_data_generators()
    source_vocab_size = len(TextData.source_vocab)
    encoding_model = Encoder(source_vocab_size, HIDDEN_SIZE)
    destination_vocab_size = len(TextData.destination_vocab)
    destination_vocab_target_size = len(TextData.destination_vocab)
    decoding_model = Decoder(destination_vocab_size, HIDDEN_SIZE, destination_vocab_target_size)
    
    # Creating Necessary directories
    create_dir(f"model_files/{model_name}")

    vocab_path = f"model_files/{model_name}/vocab.json"
    
    with open(vocab_path, "w", encoding='utf-8') as vocab_file:
                vocab_file.write(
                    json.dumps({
                        "source": TextData.source_vocab,
                        "destination": TextData.destination_vocab
                    }, ensure_ascii=False)
                )
    
    encoder_optimizer = torch.optim.Adam(encoding_model.parameters(), lr=0.01)
    decoder_optimizer = torch.optim.Adam(decoding_model.parameters(), lr=0.01)

    optimizers = [
    encoder_optimizer,
    decoder_optimizer
    ]
    loss_fun = nn.NLLLoss()
    trainer_obj = Trainer(
        loss_fun,
        optimizers,
        tensorboard=True,
        log_dir=TENSORBOARD_LOG_DIR,
        model_name=model_name,
        log_level=5,
        train_data=training_generator,
        validation_data=validate_data_generator,
        test_data=test_data_generator,
        encoder=encoding_model,
        decoder=decoding_model
        )
    
    return trainer_obj

if __name__ == "__main__":
    # import os
    # training_generator, test_data_generator, validate_data_generator = get_data_generators()
    # source_vocab_size = len(TextData.source_vocab)
    # encoding_model = Encoder(source_vocab_size, HIDDEN_SIZE)
    # destination_vocab_size = len(TextData.destination_vocab)
    # destination_vocab_target_size = len(TextData.destination_vocab)
    # decoding_model = Decoder(destination_vocab_size, HIDDEN_SIZE, destination_vocab_target_size)
    
    # model_name = 'translator_ec_dc_Adam2_telugu_clean'
    
    # vocab_path = f"model_files/{model_name}/vocab.json"
    # if model_name not in  os.listdir("model_files"):
    #     os.mkdir(f"model_files/{model_name}")
        
    # with open(vocab_path, "w", encoding='utf-8') as vocab_file:
    #     vocab_file.write(
    #         json.dumps({
    #             "source": TextData.source_vocab,
    #             "destination": TextData.destination_vocab
    #         }, ensure_ascii=False)
    #     )
    # encoder_optimizer = torch.optim.Adam(encoding_model.parameters(), lr=0.01)
    # decoder_optimizer = torch.optim.Adam(decoding_model.parameters(), lr=0.01)

    # optimizers = [
    # encoder_optimizer,
    # decoder_optimizer
    # ]
    # loss_fun = nn.NLLLoss()
    # trainer = Trainer(
    #     loss_fun,
    #     optimizers,
    #     tensorboard=True,
    #     log_dir=TENSORBOARD_LOG_DIR,
    #     model_name=model_name,
    #     log_level=5,
    #     train_data=training_generator,
    #     validation_data=validate_data_generator,
    #     test_data=test_data_generator,
    #     encoder=encoding_model,
    #     decoder=decoding_model
    #     )

    # trainer.train(EPOCHS)
    
    # configs = Configs(path="configs.yml")
    # configs.configs.update({
    #     "data":{
    #         "source_vocab_size": source_vocab_size,
    #         "destination_vocab_size": destination_vocab_size,
    #         "destination_vocab_target_size": destination_vocab_target_size,
    #         "vocab_path": vocab_path
    #     }
    # })
    # configs.dump()
    
    trainer = make_training_ready("base_model")
    trainer.train(1)
    