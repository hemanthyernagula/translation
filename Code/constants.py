"""
    This file containes all the constants required
"""

DATA_PATH = "../../../translation/Data/english_telugu/en-te/"

TENSORBOARD_LOG_DIR = "logs"


COSINE = "cosine"
DEFAULT_SCORING_FUNCTION = COSINE
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2
SOURCE_LANG = "en"
DESTINATION_LANG = "te"
BATCH_SIZE = 64
EPOCHS = 20
HIDDEN_SIZE = 256
NO_OF_LSTM_LAYERS = 1
CONFIGS_PATH = "configs.yml"

SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UKN_TOKEN = "<UNK>"
PAD_TOKEN = "<pad>"
SOS_TOKEN_INDEX = 0
EOS_TOKEN_INDEX = 1
UKN_TOKEN_INDEX = 2
PAD_TOKEN_INDEX = 3


