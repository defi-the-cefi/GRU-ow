# https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b
# https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from timeseries_data_loader import Dataset_Custom

# from Databases.pytorch_models.GRU.GRU_optimization import Optimization

#temp
import time
print('imported deps')


#%% Model Specification
# Input dimensions: batch size, sequence length, and input_dim
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, pred_len, output_dim, dropout_prob):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.output_dim = output_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob) #, bidirectional=True)

        # layer norm
        # check out https://github.com/ElektrischesSchaf/LayerNorm_GRU
        # self.layer_norm = nn.LayerNorm([batch, sequence_length, x_feature_dim])

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, self.pred_len*self.output_dim)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initializing hidden state for first input with zeros

        #   do we need requires_grad_() here
        # h0 = torch.randn(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # h02 = torch.randn(self.layer_dim//2, x.size(0), self.hidden_dim//2).requires_grad_().to(device)
        # #originally this was
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Forward propagation by passing in the input and hidden state into the model

        gru_out, hn = self.gru(x.float())#, h0) #.detach()) # out.shape = (batch_size, seq_len, hidden_size)
        print('gru_out shape: ', gru_out.shape)
        print('gru_hn shape: ', hn.shape)
        # hn is the same as gru_out[:,-1,:] i.e. only the last hidden state (last w.r.t. to number of layers AND time steps)
        # hn.shape = [layer_dim, batch_size, hidden_dim]
        out = self.Sigmoid(hn)
        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        # fc(out).shape = [layer_dim, batch_size, hidden_dim]

        # this only works for 1-hidden layer where gru_out.shape=[1, 100, self.pred_len*self.output_dim]
        print('fc out: ', out.shape)
        squeeze_out = torch.squeeze(out)
        print('squeezed: ', squeeze_out.shape)
        out = out.view(-1, self.pred_len, self.output_dim)
        print('out_view: ', out.shape)
        # out = out.view(batch_size, self.pred_len, self.output_dim)
        # print('reshaped to: ', out.shape)
        return out


class Optimization:
    def __init__(self, model, model_name, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.model_name = model_name
        self.saved_grads = []
        print('initialized optimization process')

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        # expect tensor x of shape (batch, seq_len, features)
        yhat = self.model(x)
        print('y "truth" shape : ', y.shape)
        print('yhat shape: ', yhat.shape)

        # Computes loss
        loss = self.loss_fn(y.float(), yhat)

        # Computes gradients
        loss.backward()
        avg_step_gradients = {}
        for name, weights in model.named_parameters():
            avg_step_gradients[name]=weights.grad.abs().mean()
        self.saved_grads.append(avg_step_gradients)

        print('backpropogating loss')
        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        print('optimization train step run, resetting grads')

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        print('training started')
        save_folder = r'./' + f'{datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")}'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        # TODO start saving to path dictated by file name os.path.basename(__file__)
        model_path = os.path.join(save_folder, f'{self.model_name}_{datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")}.pth')
        print('model will be saved to: ', model_path)

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                print(len(x_batch))
                print('batch size: ', batch_size)
                loss = self.train_step(x_batch.to(device), y_batch.to(device))
                batch_losses.append(loss)
                print('current batch loss: ', loss)
                print('finished process ', len(batch_losses), 'th batch of epoch: ', epoch)
            print('saving batch losses to disk')
            pd.DataFrame(batch_losses).to_csv(os.path.join(save_folder, 'batch_loss_for_epoch_'+str(epoch)+'.csv'))
            training_loss = np.mean(batch_losses)
            print('average batch loss for training epoch: ', training_loss)
            self.train_losses.append(training_loss)
            print('finished train batches, now doing val batches')
            print('average intra-batch training loss: ', self.train_losses)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    print('x_val loaded successfully')
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
            print('val batches processed')
            rolling_val_loss_avg = np.mean(self.val_losses[0 if len(self.val_losses)<10 else -10:])
            print('rolling average for past 10 val epochs: ', rolling_val_loss_avg)

            # https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py
            self.learning_rates.append(self.optimizer.param_groups[0]["lr"])
            print('learning rates hist', self.learning_rates)
            # Learning Rate scheduler update
            scheduler.step(val_loss)

            #EARLY STOPPING check
            # criteria: 2*scheduler.patience epochs with no validator error improvements
            stalled_plateau_gaps = scheduler.patience+1
            # start by verifying to we've trained for a min # of epochs
            if len(self.val_losses) >= (2 * stalled_plateau_gaps+1):
                # check if our scheduler step function just updated our learning rate
                if self.optimizer.param_groups[0]["lr"] != self.learning_rates[-1]:
                    print('new learning rate', self.optimizer.param_groups[0]["lr"])
                    # check for Early Stopping condition
                    # this check will only be true if previous 2 plateaus were of len(shceduler.patience+1)
                    if (self.learning_rates[-1] !=self.learning_rates[-(stalled_plateau_gaps+1)]) & \
                            (self.learning_rates[-(2*stalled_plateau_gaps + 1)] != self.learning_rates[-(stalled_plateau_gaps+1)]):
                        print('Early Stopping conditions met, exiting training')
                        print(self.val_losses)
                        print('validations losses')
                        save_training_loss = pd.DataFrame(self.train_losses)
                        save_training_loss.to_csv(os.path.join(save_folder,
                                                               f'training_losses_{datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")}.csv'))
                        val_losses = pd.DataFrame(self.val_losses)
                        val_losses.to_csv(os.path.join(save_folder,
                                                       f'val_losses_{datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")}.csv'))
                        return(print('early stopping, finished training'))

                    elif self.learning_rates[-(stalled_plateau_gaps + 1)] != self.learning_rates[-1]:
                        print('learning stalling, saving model, checking for convergence')
                        torch.save(self.model.state_dict(), model_path)
                        print('model successfully saved')
                        time.sleep(5)
                    else:
                        print('no convergence, keep training')
            # TODO (optional)
            #   warm restart by setting lr to our starting lr
            #   or try COSINEANNEALINGLR

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )
            print('completed training epoch: ', epoch)
            print(self.val_losses)
            print('validations losses')
        print('writing training and val losses to disk for later review')
        save_training_loss = pd.DataFrame(self.train_losses)
        save_training_loss.to_csv(os.path.join(save_folder, f'training_losses_{datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")}.csv'))
        val_losses = pd.DataFrame(self.val_losses)
        val_losses.to_csv(os.path.join(save_folder, f'val_losses_{datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")}.csv'))
        torch.save(self.model.state_dict(), model_path)
        print('model successfully saved')

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        #save_folder = r'./'
        with torch.no_grad():
            predictions = []
            values = []
            test_batch_losses = []
            evaluate_batch = 0
            for x_test, y_test in test_loader:
                print('test shapes x,y', x_test.shape, y_test.shape)
                values.append(y_test.numpy())
                self.model.eval()
                yhat = self.model(x_test.to(device))
                test_batch_loss = self.loss_fn(y_test.to(device).float(), yhat).item()
                test_batch_losses.append(test_batch_loss)
                predictions.append(yhat.detach().cpu().numpy())
                # values.append(y_test.detach().cpu().numpy())
                evaluate_batch+=1
                print('finished evaluating batch: ', evaluate_batch)
                # originally formatted as, but numpy can't be appended to cuda
                # predictions.append(yhat.to(device).detach().numpy())
                # values.append(y_test.to(device).detach().numpy())
            avg_test_loss = np.mean(test_batch_losses)
            print('average loss on test set: ', avg_test_loss)
            # TODO write preidctions, values and test_batch_losses to disk as is
            # predictions = np.vstack(predictions)
            # values = np.vstack(values)
            # print(pd.DataFrame(predictions))
            # print('writing predictions and list of truth values to disk')
            # pd.DataFrame(predictions).to_csv(os.path.join(save_folder, f'predicted_test_values_{datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")}.csv'))
            # pd.DataFrame(values).to_csv(os.path.join(save_folder, f'truth_values_for_testset_{datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")}.csv'))

        return predictions, values, test_batch_losses

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

print('finished Model and Optimization Class definitions')
