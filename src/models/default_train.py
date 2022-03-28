#imports
import torch
import os
import copy
import time
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_save_load(model,save=True,path='./models/trained_models/model.pth'):
      if save == True:
            torch.save(copy.deepcopy(model.state_dict()), os.path.abspath(path))
      else:
            model.load_state_dict(torch.load(os.path.abspath(path)))
            return model

def train_model(model, criterion, optimizer, scheduler, dataloaders,dataset_sizes,device, num_epochs=25):
    #early stopping
    the_last_loss = 100
    patience = 5
    trigger_times = 0

    #acc and loss list
    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            #print(phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels[0].to(device)
                #print(inputs)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #print(outputs)
                    #print(preds)
                    loss = criterion(outputs, labels)
                    #print(loss)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0) 
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'train':
                loss_stats['train'].append(epoch_loss)
                accuracy_stats['train'].append(epoch_acc)
            else:
                the_current_loss = epoch_loss
                loss_stats['val'].append(epoch_loss)
                accuracy_stats['val'].append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                #save current best
                if os.name == 'nt': #windows
                    path_to_model = os.path.abspath('../models/trained_models/temp_curr_best.pth')
                    model_save_load(model=model,path=path_to_model)
                else: #linux
                    home_path = os.path.expanduser('~')
                    path_to_model = f'{home_path}/scratch/code-snapshots/convolution-vs-attention/models/trained_models/temp_curr_best.pth'
                    model_save_load(model=model,path=path_to_model)
        print()
        #early stopping
        if the_current_loss > the_last_loss:
            trigger_times += 1
            print('Trigger times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                print('Best val Acc: {:4f}'.format(best_acc))

                # load best model weights
                model.load_state_dict(best_model_wts)
                return model,loss_stats,accuracy_stats
        else:
            print('Trigger times: 0')
            trigger_times = 0
        the_last_loss = the_current_loss


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,loss_stats,accuracy_stats


#Defaults
def model_default_train(model,dataloaders,dataset_sizes,device,epoch = 60):
    model.to(device=device)
    criterion = nn.CrossEntropyLoss().to(device=device)
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # Decay LR by a factor of 0.1 every 20 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    return train_model(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders,dataset_sizes,device,
                       num_epochs=epoch)
