import os
import torch
import numpy as np, time
import torch.nn.functional as F

#We are training on cpu but can easily switch to cuda if desired
device = torch.device('cpu')
#device = torch.device('cuda')
#L2 is L2 regularization strength -- generally set to 0. Others
#are fairly intuitive. Ordinal regression outputs 2 probabilities for
#3 categories.
class NN(torch.nn.Module):
    def __init__(self, l2 = 0.0000,
                 dropout = 0.2, input_dim = 180, num_outputs = 2):
        super(NN, self).__init__()
        self.l2 = l2
        self.dropout = dropout
        self.num_outputs = num_outputs
        self.fc1 = torch.nn.Linear(input_dim, 40)
        self.fc2 = torch.nn.Linear(40, 20)
        self.fc3 = torch.nn.Linear(20, 1)
        self.bn1 = torch.nn.BatchNorm1d(40)
        self.bn2 = torch.nn.BatchNorm1d(20)
        self.o_weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.o_bias = torch.nn.Parameter(torch.zeros(self.num_outputs))

    #Forward pass.
    def forward(self, x, training = True):
        x = self.bn1(F.elu(self.fc1(x)))
        if self.dropout != 0:
            x = F.dropout(x, p=self.dropout, training=training)
        x = self.bn2(F.elu(self.fc2(x)))
        if self.dropout != 0:
            x = F.dropout(x, p=self.dropout, training=training)
        #This is the unique part. The fully connected NN outputs a single
        #value, which is then added to the same number of "bias" terms
        #or thresholds as there are outputs. This is the key to ordinal
        #regression, since it forces the model to treat the categories as
        #ordinal in nature.
        x = self.fc3(x)
        output = self.o_weight*x + self.o_bias
        return torch.sigmoid(output)

    #Custom loss function. We could use a Pytorch loss function
    #but this gives us more control. Basically just binary
    #cross entropy loss with weighting. We do not actually use
    #task weights but this was left as an option in case ever needed.
    def nll(self, ypred, ytrue, task_weights, weights=None):
        lossterm1 = -torch.log(ypred)*ytrue
        lossterm2 = -torch.log(torch.clamp(1-ypred, min=1*10**-10))*(1-ytrue)
        loss = torch.sum((lossterm1 + lossterm2), dim=1)
        if weights is not None:
            loss *= weights
        return torch.mean(loss) + self.o_weight*0.001

    #Call this function to train the model.
    def trainmod(self, input_data, epochs=30, minibatch=100, track_loss = True,
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
        #We do not actually use task weights at present, so this if statement
        #is merely here in case we decide to incorporate those.
        if use_weights == True:
            task_scores = [np.sum(x[np.argwhere(x[:,180]==1).flatten(),185]),
                           np.sum(x[np.argwhere(x[:,181]==1).flatten(),185])]
            task_scores = np.asarray(task_scores)
            task_weights = np.maximum(task_scores, np.sum(x[:,185]) - task_scores)
        else:
            task_scores = np.sum(x[:,180:(180+self.num_outputs)],0)
            task_weights = np.maximum(task_scores, x.shape[0] - task_scores)
        max_score = np.max(task_weights)
        task_weights = torch.from_numpy(np.sqrt(task_weights) /
                                        np.sqrt(max_score)).float()
        losses = []
        for i in range(0, epochs):
            next_epoch, current_position = False, 0
            #Loop over the data for the user selected number of epochs
            while(next_epoch == False):
                #On each round, if we haven't reached the end of the data, select a minibatch
                #and update our position.
                if current_position > x.shape[0]:
                    break
                elif (current_position + minibatch) > (x.shape[0]-2):
                    #The one-hot encoded sequence is columns 0:180 of the input matrix;
                    #the category assignments are columns 180:182; the weights are in
                    #column 185.
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
                #Do forward pass, clamp probabilities to avoid zeros
                y_pred = self.forward(x_mini).clamp(min=1*10**-10)
                if use_weights == True:
                    loss = self.nll(y_pred, y_mini,
                                    task_weights=task_weights,
                                    weights=mini_weights)
                else:
                    loss = self.nll(y_pred, y_mini,
                                    task_weights=task_weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                time.sleep(0.005)
            if track_loss == True:
                print('Current loss: %s'%loss.item())
                losses.append(loss.item())
        if track_loss == True:
            return losses

    #This function is the same as forward BUT instead of returning class probabilities
    #it returns the latent variable.
    def extract_hidden_rep(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x[:,:180]).float()
            #self.eval is necessary to get correct behavior from batchnorm.
            self.eval()
            x = self.bn1(F.elu(self.fc1(x)))
            x = self.bn2(F.elu(self.fc2(x)))
            x = self.o_weight*F.elu(self.fc3(x))
        return x

    #This function returns both predicted class (second returned object) and predicted
    #class probabilities (first returned class object) for an input data matrix.
    def predict(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x[:,:180]).float()
            #self.eval is necessary to get correct behavior from batchnorm.
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
