import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        ''' Initialize the layers of this model.'''
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Embedding layer that turns words into a vector of a specified size
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_size
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=drop_prob, batch_first=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # Linear layer that maps the hidden state output dimension 
        # to the number of vocabulary we want as output, vocab_size
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        # Initialize the weights
        self.init_weights()
    
    def init_weights(self):
        ''' Initialize weights for the linear layer '''
        # Set bias tensor to all zeros
        self.linear.bias.data.fill_(0)
        # FC weights as random uniform
        self.linear.weight.data.uniform_(-1, 1)
        
    def init_states(self, batch_size):
        ''' At the start of training, we need to initialize a hidden state and a cell state;
           there will be none because the states is formed based on perviously seen data.
           So, this function defines the states with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros(self.num_layers , batch_size , self.hidden_size).to(device),
                torch.zeros(self.num_layers , batch_size , self.hidden_size).to(device))
    
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.'''
        # Initialize the hidden/cell state
        self.states = None
        # self.batch_size = features.shape[0]
        # self.states = self.init_states(self.batch_size)
        
        # Create embedded word vectors for each word in a sentence from caption
        embeds = self.word_embed(captions[:, :-1]) # except last word, "<end>".
        embeds = torch.cat((features.unsqueeze(dim = 1), embeds), dim = 1) # 1st:image feature vector, 2nd~:word　feature vectorer
        
        # Get the output and statea by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and states
        lstm_out, self.states = self.lstm(embeds , self.states) # states = (hidden_state' history, cell_state' history)
        
        # Get the scores for the most likely tag for a word
        
        # Please remove the dropout layer and use the parameter of LSTM layer 
        # to add dropout.. I would suggest you not to use a dropout layer 
        # in between LSTM and Linear layer..
        # outputs = self.linear(self.dropout(lstm_out)) 
        
        outputs = self.linear(lstm_out)
        
        # Cross entropy of loss function is equivalent to applying LogSoftmax on an input, followed by NLLLoss.
        # outputs = F.log_softmax(outputs, dim=1)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentences_idx = [] # indx
        # batch = inputs.shape[0]
        # states = self.init_states(batch)
        
        # 1st  : inputs is image feature vector from CNN, output is fisrt word index.
        # 2nd~ : inputs is previous output word feature vevtor, output is next word index which predicted by the inputs and states.
        
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            outputs = self.linear(lstm_out)
            predicted_word_idx = torch.argmax(outputs, dim=-1) # indx
            sentences_idx.append(int(predicted_word_idx.cpu().detach().numpy()[0][0]))
            inputs = self.word_embed(predicted_word_idx)
        
            if int(predicted_word_idx.cpu().detach().numpy()[0][0]) == 1:
                break
            
        return sentences_idx
