import os
import torch
import numpy as np, time
import torch.nn.functional as F

device = torch.device('cpu')
#device = torch.device('cuda')
#A fully connected neural network that predicts enrichment going from RH02 to RH03
#as a possible alternative to ordinal regression.
class enrichment_nn(torch.nn.Module):
    def __init__(self, l2 = 0.0000,
                 dropout = 0.2, input_dim = 180, num_outputs = 1):
        super(enrichment_nn, self).__init__()
        self.l2 = l2
        self.dropout = dropout
        self.num_outputs = num_outputs
        self.fc1 = torch.nn.Linear(input_dim, 40)
        self.fc2 = torch.nn.Linear(40, 20)
        self.fc3 = torch.nn.Linear(20, num_outputs)
        self.bn1 = torch.nn.BatchNorm1d(40)
        self.bn2 = torch.nn.BatchNorm1d(20)


    def forward(self, x, training = True):
        x = self.bn1(F.elu(self.fc1(x)))
        if self.dropout != 0:
            x = F.dropout(x, p=self.dropout, training=training)
        x = self.bn2(F.elu(self.fc2(x)))
        if self.dropout != 0:
            x = F.dropout(x, p=self.dropout, training=training)
        x = self.fc3(x)
        return x.squeeze()


    def trainmod(self, input_data, epochs=20, minibatch=100, track_loss = True,
          use_weights = True, optim = 'adam', lr=0.01):
        x = input_data
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr,
                                 weight_decay = self.l2)


        loss_fn = torch.nn.MSELoss()
        for i in range(0, epochs):
            next_epoch, current_position = False, 0
            while(next_epoch == False):
                if current_position > x.shape[0]:
                    break
                elif (current_position + minibatch) > (x.shape[0]-2):
                    x_mini = torch.from_numpy(x[current_position:,
                                                :180]).float()
                    y_mini = torch.from_numpy(x[current_position:, 180]).float()
                    current_position += minibatch
                    next_epoch=True
                else:
                    x_mini = torch.from_numpy(x[current_position:current_position+minibatch,
                                                :180]).float()
                    y_mini = torch.from_numpy(x[current_position:current_position+minibatch,
                                                180]).float()
                    current_position += minibatch
                y_pred = self.forward(x_mini)
                loss = loss_fn(y_pred, y_mini)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                time.sleep(0.005)
            if track_loss == True:
                print('Current loss: %s'%loss.item())


    def predict(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x[:,:180]).float()
            self.eval()
            return 0, self.forward(x, training=False).numpy()
