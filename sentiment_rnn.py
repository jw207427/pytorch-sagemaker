import argparse
import json
import logging
import os
import sys
import io
import tempfile

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

vocab_size = 74073
output_size = 1 
embedding_dim = 400 
hidden_dim = 256
n_layers = 2

# check if GPU is available
use_cuda=torch.cuda.is_available()

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        lstm_out = lstm_out[:, -1, :] # getting the last time step output
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (use_cuda):
            hidden = (weight.new(self.n_layers, batch, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch, self.hidden_dim).zero_())
        
        return hidden

def create_data_loader(args):
    logger.info('Start loading data...')

    # load training data
    train = np.loadtxt(f'{args.train}/train.txt', delimiter=',')
    valid = np.loadtxt(f'{args.valid}/valid.txt', delimiter=',')
    
    train_x, train_y = train[:, :-1],train[:, -1:].reshape(train.shape[0])
    valid_x, valid_y = valid[:, :-1],valid[:, -1:].reshape(valid.shape[0])
    
    logger.info(f'training data size is {train_x.shape}')
    logger.info(f'validation data size is {valid_x.shape}')

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x),
                               torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(valid_x),
                               torch.from_numpy(valid_y))
    
    # make sure the SHUFFLE your training data
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=args.batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True,
                              batch_size=args.batch_size)
    
    return train_loader, valid_loader

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)
    
def train(args, train_loader, valid_loader):
    
    logger.info('Start training loop...')
    
    net = SentimentRNN(args.vocab_size, 
                       args.output_size, 
                       args.embedding_dim, 
                       args.hidden_dim, 
                       args.n_layers)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    if use_cuda:
        net.cuda()
        
    net.train()
    counter = 0
    clip = 5
    print_every = 100
    
    # train for some number of epochs
    for e in range(args.epochs):
        # initialize hidden state
        h = net.init_hidden(args.batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if(use_cuda):
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(args.batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if(use_cuda):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                net.train()
                
                logger.info(f"Epoch: {e+1}/{args.epochs}...")
                logger.info(f"Step: {counter}...")
                logger.info(f"Loss: {loss.item()}...")
                logger.info(f"Val_Loss: {np.mean(val_losses)}")
                
    logger.info("Training complete successfully....")

    return save_model(net, args.model_dir)


def model_fn(model_dir):

    model = SentimentRNN(vocab_size, output_size, embedding_dim,hidden_dim,n_layers)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    
    if use_cuda:
        model.cuda()
    
    return model

def input_fn(request_body, request_content_type):
    print('custom input function.....')
    
    f = io.BytesIO(request_body)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())
    print(tfile.name)
    
    test = np.loadtxt(tfile.name, delimiter=',')
    
    test_x, test_y = test[:, :-1],test[:, -1:].reshape(test.shape[0])
    
    print(f'Input numpy array size {test_x.shape}....')
    return [test_x, test_y]

def predict_fn(data, model):
    print('custom predict function.....')
    
    input_data = torch.from_numpy(data[0])
    
    print(f"Input data shape {data[0].shape} and label shape {data[1].shape}...")
    
    h = model.init_hidden(input_data.shape[0])
    with torch.no_grad():
        if use_cuda:
            input_data = input_data.cuda()
        model.eval()

        output, h = model(input_data, h)
        
        pred = torch.round(output.squeeze())
        
        pred = np.squeeze(pred.numpy()) if not use_cuda else np.squeeze(pred.cpu().numpy())
        
        print(f"Prediction data shape {pred.shape}....")

    final_output = np.array((data[1], pred)).T  
    return final_output

    
def output_fn(output_batch, accept='application/json'):
    print('custom output function.....')
    res = []
    print('output list length')
    print(len(output_batch))
    for output in output_batch:
        res.append({'label':output[0], 'pred':output[1]})
    
    return json.dumps(res)

if __name__ == "__main__":
    
    logger.info('Parsing command-line arguments...')

    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="The number of steps to use for training."
    )  
    
    parser.add_argument('--vocab_size', type=int, default=vocab_size)
    parser.add_argument("--output_size", type=int, default=output_size)
    parser.add_argument("--embedding_dim", type=int, default=embedding_dim)
    parser.add_argument("--hidden_dim", type=int, default=hidden_dim)
    parser.add_argument("--n_layers", type=int, default=n_layers)
    
    # Data directories
    parser.add_argument('--train', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--valid', type=str,
                        default=os.environ.get('SM_CHANNEL_VALID'))
    # model directory
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    
    args = parser.parse_args()

    # Load the dataset
    train_loader, valid_loader = create_data_loader(args)
    
    # Train the LSTM model
    train(args, train_loader, valid_loader)
    
    logger.info('Training complete...')
    
    