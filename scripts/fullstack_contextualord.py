import os
import torch
import numpy as np, time
import torch.nn.functional as F

device = torch.device('cpu')
#device = torch.device('cuda')
#Default number of outputs is 2 since we have a 3-class problem and are
#using ordinal regression.
class NN(torch.nn.Module):
    def __init__(self, l2 = 0.0000,
                 dropout = 0.3, input_dim = 180, num_outputs = 2):
        super(NN, self).__init__()
        self.l2 = l2
        self.dropout = dropout
        self.num_outputs = num_outputs
        self.fc1 = torch.nn.Linear(input_dim, input_dim)
        self.fc2 = torch.nn.Linear(input_dim, input_dim)
        self.fc3 = torch.nn.Linear(input_dim, input_dim)
        self.bn1 = torch.nn.BatchNorm1d(input_dim)
        self.bn2 = torch.nn.BatchNorm1d(input_dim)
        self.o_weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.o_bias = torch.nn.Parameter(torch.tensor([0.0, 0.0]))


    def forward(self, x, training = True):
        x1 = self.bn1(F.elu(self.fc1(x)))
        if self.dropout != 0:
            x1 = F.dropout(x1, p=self.dropout, training=training)
        x1 = self.bn2(F.elu(self.fc2(x1)))
        if self.dropout != 0:
            x1 = F.dropout(x1, p=self.dropout, training=training)
        #Here's the key difference between this and the ordinal regression
        #network. In this contextual regression step, we multiply the output
        #of the last layer times the input and sum. Essentially the
        #network generates linear weights for the input through a nonlinear
        #function of the input.
        x = torch.sum(self.fc3(x1)*x, dim=1).unsqueeze(1)
        output = self.o_weight*x + self.o_bias
        return torch.sigmoid(output)
    #This function will return the sequence-specific weights for an input
    #i.e. the feature importances. Feature importances are sequence-
    #specific when using contextual regression, so we will need to average
    #across > 1 sequence.
    def generate_feat_importances(self, x):
        with torch.no_grad():
            self.eval()
            x = torch.from_numpy(x[:,:180]).float()
            x1 = self.bn1(F.elu(self.fc1(x)))
            x1 = self.bn2(F.elu(self.fc2(x1)))
            return self.fc3(x1).numpy()

    def nll(self, ypred, ytrue, task_weights, weights=None):
        lossterm1 = -torch.log(ypred)*ytrue
        lossterm2 = -torch.log(torch.clamp(1-ypred, min=1*10**-10))*(1-ytrue)
        loss = torch.sum((lossterm1 + lossterm2), dim=1)
        if weights is not None:
            loss *= weights
        return torch.mean(loss)

    #Same as the ordinal regression trainmod function...
    def trainmod(self, input_data, epochs=20, minibatch=100, track_loss = True,
          use_weights = False, optim = 'adam', lr=0.01, momentum = 0.1):
        x = input_data
        self.train()
        if optim == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr,
                                 weight_decay = self.l2)
        elif optim == 'rms':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=lr,
                                 weight_decay = self.l2)
        elif optim == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr,
                                 weight_decay = self.l2, momentum=momentum,
                                    nesterov=True)
        task_weights=None

        losses = []
        for i in range(0, epochs):
            next_epoch, current_position = False, 0
            while(next_epoch == False):
                if current_position > x.shape[0]:
                    break
                elif (current_position + minibatch) > (x.shape[0]-2):
                    x_mini = torch.from_numpy(x[current_position:,
                                                :180]).float()
                    y_mini = torch.from_numpy(x[current_position:,
                                                180:(180+self.num_outputs)]).float()
                    mini_weights = torch.from_numpy(x[current_position:,
                                                      185]).float()
                    current_position += minibatch
                    next_epoch = True
                else:
                    x_mini = torch.from_numpy(x[current_position:current_position+minibatch,
                                                :180]).float()
                    y_mini = torch.from_numpy(x[current_position:current_position+minibatch,
                                                180:(180+self.num_outputs)]).float()
                    mini_weights = torch.from_numpy(x[current_position:current_position+minibatch,
                                                      185]).float()
                    current_position += minibatch
                y_pred = self.forward(x_mini).clamp(min=1*10**-10)
                if use_weights == True:
                    loss = self.nll(y_pred, y_mini, task_weights, mini_weights)
                else:
                    loss = self.nll(y_pred, y_mini, task_weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                time.sleep(0.002)
            if track_loss == True:
                print('Current loss: %s'%loss.item())
                losses.append(loss.item())
        if track_loss == True:
            return losses

    #This function returns "score" latent values for input sequences.
    def extract_hidden_rep(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x[:,:180]).float()
            self.eval()
            x1 = self.bn1(F.elu(self.fc1(x)))
            x1 = self.bn2(F.elu(self.fc2(x1)))
            x = torch.sum(self.fc3(x1)*x, dim=1)
            output = self.o_weight*x
        return output

#Use the predict function to make predictions with a trained model.
#Note that for classification, it returns both class probabilities AND
#predicted categories.
    def predict(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x[:,:180]).float()
            self.eval()
            probs = self.forward(x, training=False).numpy()
            class_pred = []
            for i in range(0, probs.shape[0]):
                if probs[i,1] > 0.5:
                    class_pred.append(2)
                elif probs[i,0] > 0.5:
                    class_pred.append(1)
                else:
                    class_pred.append(0)
        return probs, np.asarray(class_pred)
