import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h as 0 if these values are not given.                          #
        ############################################################################
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        dtype = torch.FloatTensor
        w0 = torch.FloatTensor(hidden_size, hidden_size).type(dtype)
        self.W_hh = Variable(w0, requires_grad=True)
        self.W_ih = Variable(torch.rand(self.input_size, self.hidden_size), requires_grad=True)
        self.b_ih = Variable(torch.zeros(self.hidden_size), requires_grad=True)
        self.fc = nn.Linear(hidden_size, input_size)

        # Is it necessary to implement the following way?
        # self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_dim, output_size)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        seq_len, batch_size, input_size = x.shape
        for i in range(batch_size):
            x_batch = x[:, i, :]
            h_seq = x_batch @ self.W_ih
            if h:
                nr_layers, batch_size, hidden_size = h.shape
                for j in range(nr_layers):
                    h_seq += h[j, i, :] @ self.W_hh
            h_seq = torch.tanh(h_seq)
        h = h_seq[seq_len-1]


        # Is it necessary to implement the following way?
        # batch_size = x.size(0)
        # r_out, hidden = self.rnn(x, hidden)
        # # shape output to be (batch_size*seq_length, hidden_dim)
        # r_out = r_out.view(-1, self.hidden_dim)
        # output = self.fc(r_out)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , h

class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
        ############################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes      #
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h and c as 0 if these values are not given.                    #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
            
    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq=[]
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        pass

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , (h, c)
    

class RNN_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()
        ############################################################################
        #  TODO: Build a RNN classifier                                            #
        ############################################################################
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def forward(self, x):
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        pass

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

class LSTM_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28 , hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        ############################################################################
        #  TODO: Build a LSTM classifier                                           #
        ############################################################################
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def forward(self, x):
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        pass

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
        