import matplotlib.pyplot as plt
import numpy as np
from data_loader import TextData
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
import torch
import json


font = FontProperties(fname="/home/hemanth/fonts/nirmala-ui/Nirmala.ttf")
def plot_att_wts(input_, output, attentions, source_vocab, destination_vocab, epoch=0):
    
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    cax = ax.matshow(np.array(attentions))
    fig.colorbar(cax)
    ax.set_xticklabels(['']+[source_vocab[x] for x in input_[0]])
    ax.set_yticklabels(['test']+[destination_vocab[x] for x in output], fontproperties=font)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # plt.show()
    plt.savefig(f'attention_wts/{epoch}_ats.png')
    return fig
    
    
if __name__ == "__main__":
    outputs = [i for i in range(50, 60)]
    attentions = [torch.randn(8,).squeeze().cpu().detach().numpy() for i in range(10)]
    inputs_ = list(torch.randint(0,100, size=(1,8)).numpy())
    
    vocab = json.loads(open("Code/model_files/base_model_v1.5.2/vocab.json").read())
    source_vocab = vocab.get("source")
    destination_vocab = vocab.get("destination")
    plot_att_wts(inputs_[0], outputs, attentions, source_vocab, destination_vocab)
    