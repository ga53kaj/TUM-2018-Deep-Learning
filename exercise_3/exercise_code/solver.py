from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################

        log_count = log_nth
        for epoch in range(1, num_epochs + 1):
            ###################
            # train the model #
            ###################
            model.train()
            for i, batch in enumerate(train_loader):
                data, target = batch
                # clear the gradients of all optimized variables
                optim.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model.forward(data)
                # calculate the batch loss
                loss = 0
                for s in range(data.size()[0]):
                    loss += self.loss_func(output[s].view(23, -1).transpose(1, 0),
                                           target[s].view(-1))
                # loss = self.loss_func(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optim.step()
                # update training loss
                train_loss = loss.item()
                self.train_loss_history.append(train_loss)

                if log_count == 0:
                    print("[Iteration %d/%d] TRAIN loss: %.3f" % (i, iter_per_epoch, train_loss))
                    log_count = log_nth

            # Train acc / loss (last batch)
            pred = torch.argmax(output, dim=1)
            train_acc = torch.sum(pred == target).item() / train_loader.batch_size
            self.train_acc_history.append(train_acc)

            # Validation acc / loss (all val batches)
            # val_acc_list = []
            # val_loss_list = []
            #
            # ######################
            # # validate the model #
            # ######################
            # model.eval()
            # for i, batch in enumerate(val_loader):
            #     data, target = batch
            #     # forward pass: compute predicted outputs by passing inputs to the model
            #     output = model.forward(data)
            #     # # calculate the batch loss
            #     # loss = self.loss_func(output, target)
            #     # # update average validation loss
            #     # valid_loss = loss.item()
            #     pred = torch.argmax(output, dim=1)
            #     val_acc_list.append(torch.sum(pred == target).item())
            #     val_loss_list.append(self.loss_func(output, target).item())
            #
            # val_acc = np.mean(val_acc_list) / val_loader.batch_size
            # val_loss = np.mean(val_loss_list)
            # self.val_acc_history.append(val_acc)
            # self.val_loss_history.append(val_loss)

            # print training/validation statistics
            print("[Epoch %d/%d] TRAIN acc/loss: %.6f/%.6f" % (epoch, num_epochs, train_acc, train_loss))
            # print("[Epoch %d/%d] VAL   acc/loss: %.6f/%.6f" % (epoch, num_epochs, val_acc, val_loss))

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
