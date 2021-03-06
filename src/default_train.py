# imports
import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb
from torch.optim import lr_scheduler
from torch.utils import model_zoo
from tqdm import tqdm


# Save and Load Model function
def model_save_load(model, save=True, path='./models/trained_models/model.pth'):
    if save == True:
        torch.save(copy.deepcopy(model.state_dict()), os.path.abspath(path))
    else:
        model.load_state_dict(torch.load(os.path.abspath(path)))
        return model


# Train function for melanoma dataset
def train_model_m(model, args, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs):
    # early stopping
    best_loss = 100
    patience = 22
    trigger_times = 0

    # acc and loss list
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
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.type(torch.LongTensor).to(device)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
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
                wandb.log({"epoch": epoch, "Train epoch_loss": epoch_loss, "Train epoch_acc": epoch_acc})
            else:
                the_current_loss = epoch_loss
                loss_stats['val'].append(epoch_loss)
                accuracy_stats['val'].append(epoch_acc)
                wandb.log({"epoch": epoch, "Valid epoch_loss": epoch_loss, "Valid epoch_acc": epoch_acc})

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # save current best
                # if os.name == 'nt':  # windows
                #     path_to_model = os.path.abspath('../models/trained_models/temp_curr_best.pth')
                #     model_save_load(model=model, path=path_to_model)
                # else:  # linux
                #     home_path = os.path.expanduser('~')
                #     path_to_model = f'{home_path}/scratch/code-snapshots/convolution-vs-attention/models/trained_models/temp_curr_best.pth'
                #     model_save_load(model=model, path=path_to_model)
                path_to_model = os.path.join(args.trained_model_dir, 'temp_curr_best.pth')
                model_save_load(model=model, path=path_to_model)
        print()
        # early stopping
        if the_current_loss > best_loss:
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
                return model, loss_stats, accuracy_stats
        else:
            best_loss = the_current_loss
            print('Trigger times: 0')
            trigger_times = 0

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_stats, accuracy_stats


# Default training pipeline for melanoma dataset
def model_default_train_m(model, args, dataloaders, dataset_sizes, device, epoch):
    model.to(device=device)
    class_weights = torch.FloatTensor([0.187, 0.813]).to(device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device=device)
    # Observe that all parameters are being optimized
    optimizer_ft = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

    # Decay LR by a factor of 0.1 every 20 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
    wandb.log({"lr": optimizer_ft.param_groups[0]["lr"]})
    return train_model_m(model, args, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, device,
                         num_epochs=epoch)


# Train function for SIN dataset
def train_model(model, args, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs):
    # early stopping
    best_loss = 100
    patience = 5
    trigger_times = 0

    # acc and loss list
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
            # print(phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels[0].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
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
                wandb.log({"epoch": epoch, "Train epoch_loss": epoch_loss, "Train epoch_acc": epoch_acc})
            else:
                the_current_loss = epoch_loss
                loss_stats['val'].append(epoch_loss)
                accuracy_stats['val'].append(epoch_acc)
                wandb.log({"epoch": epoch, "Valid epoch_loss": epoch_loss, "Valid epoch_acc": epoch_acc})

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # save current best
                # if os.name == 'nt':  # windows
                #     path_to_model = os.path.abspath('../models/trained_models/temp_curr_best.pth')
                #     model_save_load(model=model, path=path_to_model)
                # else:  # linux
                #     home_path = os.path.expanduser('~')
                #     path_to_model = f'{home_path}/scratch/code-snapshots/convolution-vs-attention/models/trained_models/temp_curr_best.pth'
                #     model_save_load(model=model, path=path_to_model)
                path_to_model = os.path.join(args.trained_model_dir, 'temp_current_best.pth')
                model_save_load(model=model, path=path_to_model)
        print()
        # early stopping
        if the_current_loss > best_loss:
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
                return model, loss_stats, accuracy_stats
        else:
            best_loss = the_current_loss
            print('Trigger times: 0')
            trigger_times = 0

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_stats, accuracy_stats


# Default training pipeline for SIN dataset
def model_default_train(model, args, dataloaders, dataset_sizes, device, epoch):
    model.to(device=device)
    criterion = nn.CrossEntropyLoss().to(device=device)
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # Decay LR by a factor of 0.1 every 20 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
    wandb.log({"lr": optimizer_ft.param_groups[0]["lr"]})
    return train_model(model, args, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, device,
                       num_epochs=epoch)


# load pretrained resnet50 models pretrained on SIN and IN
def load_model(model_name):
    model_urls = {
        'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
        'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
        'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
    }

    if "resnet50" in model_name:
        print("Using the ResNet50 architecture.")
        model = torchvision.models.resnet50(pretrained=False)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = model_zoo.load_url(model_urls[model_name])


    elif "alexnet" in model_name:
        print("Using the AlexNet architecture.")

        # download model from URL manually and save to desired location
        filepath = "./alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar"

        assert os.path.exists(
            filepath), "Please download the AlexNet model yourself from the following link and save it locally: https://drive.google.com/drive/u/0/folders/1GnxcR6HUyPfRWAmaXwuiMdAMKlL1shTn"

        model = torchvision.models.alexnet(pretrained=False)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        checkpoint = torch.load(filepath)
    else:
        raise ValueError("unknown model architecture.")

    model.load_state_dict(checkpoint["state_dict"])
    return model
