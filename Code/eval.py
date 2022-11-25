import json

import torch
from loguru import logger

from constants import (CONFIGS_PATH, EOS_TOKEN, EOS_TOKEN_INDEX, PAD_TOKEN,
                        PAD_TOKEN_INDEX, SOS_TOKEN, SOS_TOKEN_INDEX, UKN_TOKEN,
                        UKN_TOKEN_INDEX, HIDDEN_SIZE)
from data_loader import LoadAndData
from models.model import LSTMDecoder, LSTMEncoder
from utils.load_configs import Configs


class RunInference:
    def __init__(self, encoder, decoder, source_vocab, destination_vocab):
        self.encoding_model = encoder
        self.decoding_model = decoder
        self.source_vocab = source_vocab
        self.destination_vocab = destination_vocab

    def predict(self, data_point):
        """_summary_

        Args:
            data_point (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = data_point

        pred_out = []
        logger.info(x)
        self.encoding_model.eval()
        self.decoding_model.eval()
        with torch.no_grad():
            enc_out = torch.zeros(20, 5)
            logger.info(enc_out.shape)
            for e_w in range(1, x.shape[0]):
                # Iterating each word and tringing the encoder
                inp = x[e_w].view(1)
                encoding_out, encoding_h_n, encoding_c_n = self.encoding_model(inp)
                encoding_out = encoding_out.squeeze()
                # logger.info(f"{encoding_out, encoding_out.squeeze().shape}")
                enc_out[e_w] = encoding_out[0]

            d_inp = torch.tensor(SOS_TOKEN_INDEX).view(1)
            d_hid = encoding_h_n
            d_c_n = encoding_c_n
            
            logger.info(f"{d_hid.shape, d_c_n.shape, d_inp.shape, d_inp}")
            for d_w in range(20):
                d_hid = d_hid.squeeze()
                d_c_n = d_c_n.squeeze()
                d_out, d_hid, d_c_n = self.decoding_model(d_inp, d_hid, d_c_n)
                topv, topi = d_out.data.topk(1)
                topi = topi.squeeze(1)
                logger.info(f"inp : {d_inp} topi : {topi, topi.shape}")
                d_inp = topi.detach()
                pred_out.append(topi)
        
        return pred_out
    
    def run(self, text):
        source_words = []
        logger.info(f"original text : {text}")
        text = LoadAndData.clean_sent(text)
        logger.info(f"cleaned text : {text}")
        
        text = f"{SOS_TOKEN} {text} {EOS_TOKEN}"
        for word in text.split(" "):
            try:
                source_words.append(self.source_vocab.index(word))
            except ValueError:
                source_words.append(UKN_TOKEN_INDEX)
                
        
        logger.info(f"source_words : {source_words}")
        source_vect = torch.tensor(source_words)
        
        logger.info(f"source_vect shape : {source_vect.shape}")
        pred_indexes = self.predict(source_vect)
        
        pred_words = [self.make_sentance_from_index(i) for i in pred_indexes]
        
        return ' '.join(pred_words)
    
    def make_sentance_from_index(self, indexes):
        final_sentance = ""
        # for index in indexes[0].numpy():
        try:
            final_sentance += f" {self.destination_vocab[indexes]}"
        except ValueError:
            if indexes is PAD_TOKEN_INDEX:
                final_sentance += PAD_TOKEN
            else:
                final_sentance += " <unk>"
        
        return final_sentance
    

if __name__ == "__main__":
    configs = Configs(path=CONFIGS_PATH).configs
    vocab = json.load(
        open(configs.get("data").get("vocab_path"), "r", encoding="utf-8")
        )
    
    source_vocab = vocab.get("source")
    destination_vocab = vocab.get("destination")
    
    source_vocab_size = len(source_vocab)
    destination_vocab_size = len(destination_vocab)
    destination_vocab_target_size = len(destination_vocab)
    
    
    encoder = LSTMEncoder(source_vocab_size, HIDDEN_SIZE)
    decoder = LSTMDecoder(destination_vocab_size, HIDDEN_SIZE, destination_vocab_target_size)
    
    base_name = "base_model_v1/19/"
    encoder.load_state_dict(torch.load(f"./model_files/{base_name}encoder.pt"))
    decoder.load_state_dict(torch.load(f"./model_files/{base_name}decoder.pt"))
    
    # encoder.eval()
    # decoder.eval()
    text = "<sos> australian batsman david warner . <pad> <eos>"
    eval = RunInference(encoder, decoder, source_vocab, destination_vocab)
    print(eval.run(text))
    