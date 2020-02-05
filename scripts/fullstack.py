import os
import torch
import numpy as np, time
import torch.nn.functional as F

device = torch.device('cpu')
#device = torch.device('cuda')
#A fairly normal, fairly boring fully connected neural network
#with the same dimensions as the ordinal regression network but
#generating class probabilities using softmax activation across
#a vector of dimension num_outputs.
class NN(torch.nn.Module):
    def __init__(self, l2 = 0.0000,
                 dropout = 0.2, input_dim = 180, num_outputs = 3):
        super(NN, self).__init__()
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
        return F.softmax(x, dim=1)


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
        if use_weights == True:
            task_scores = [np.sum(x[np.argwhere(x[:,180]==0).flatten(),182]),
                           np.sum(x[np.argwhere(x[:,180]==1).flatten(),182]),
                           np.sum(x[np.argwhere(x[:,180]==2).flatten(),182])]
            task_weights = np.asarray(task_scores)
        else:
            task_weights = np.asarray([np.argwhere(x[:,180]==0).shape[0],
                                   np.argwhere(x[:,180]==1).shape[0],
                                        np.argwhere(x[:,180]==2).shape[0]])
        task_weights = np.max(task_weights) / task_weights
        task_weights = torch.from_numpy(task_weights).float()
        print(task_weights)
        loss_fn = torch.nn.NLLLoss(reduction='none')
        losses = []
        for i in range(0, epochs):
            next_epoch, current_position = False, 0
            while(next_epoch == False):
                if current_position > x.shape[0]:
                    break
                elif (current_position + minibatch) > (x.shape[0]-2):
                    x_mini = torch.from_numpy(x[current_position:,
                                                :180]).float()
                    y_mini = torch.from_numpy(x[current_position:, 180]).long()
                    mini_weights = torch.from_numpy(x[current_position:,
                                                      182]).float()
                    current_position += minibatch
                    next_epoch=True
                else:
                    x_mini = torch.from_numpy(x[current_position:current_position+minibatch,
                                                :180]).float()
                    y_mini = torch.from_numpy(x[current_position:current_position+minibatch,
                                                180]).long()
                    mini_weights = torch.from_numpy(x[current_position:current_position+minibatch,
                                                      182]).float()
                    current_position += minibatch
                y_pred = torch.log(self.forward(x_mini).clamp(min=1*10**-10))
                if use_weights == True:
                    loss = loss_fn(y_pred, y_mini) * mini_weights
                else:
                    loss = loss_fn(y_pred, y_mini)
                loss = torch.mean(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                time.sleep(0.005)
            if track_loss == True:
                print('Current loss: %s'%loss.item())
                losses.append(loss.item())
        if track_loss == True:
            return losses

    def predict(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x[:,:180]).float()
            self.eval()
            probs = self.forward(x, training=False).numpy()
            class_pred = np.argmax(probs, axis=1)
            return probs, np.asarray(class_pred)
