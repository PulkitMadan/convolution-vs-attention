from unittest import result
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os

from sklearn.metrics import classification_report, confusion_matrix
#class mapping functions
from data.load_data import class_16_listed, map207_to_16names, map207_to_16, mapping_207_reverse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#define path for visualizations
if os.name == 'nt': #windows
    fig_path = os.path.abspath(f'./visualization')
else: #linux
    home_path = os.path.expanduser('~')
    fig_path = f'{home_path}/scratch/code-snapshots/convolution-vs-attention/src/visualization'

#visualize training/val loss and accuracy
def visualize_loss_acc(loss_stats, accuracy_stats,name='loss_acc_plot'):
    # Create dataframes
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    # Plot the dataframes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
    sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
    fig.suptitle(name)
    plt.savefig(f'{fig_path}/{name}.png')


def accuracy_topk(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

#Classification report and confusion matrix evaluation on melanoma dataset
def eval_test(model, dataloaders,dataset_sizes):
    model.to(device=device)
    y_pred_list = []
    y_true_list = []
    running_corrects = 0
    with torch.no_grad():
        model.eval()
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            # batch_size = len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_pred_list = np.concatenate((y_pred_list, preds.cpu().numpy()), axis=None)
            y_true_list = np.concatenate((y_true_list, labels.cpu().numpy()), axis=None)
            #print(y_pred_list)
            #print(y_true_list)
            running_corrects += torch.sum(preds == labels.data)

        # y_pred_list = [i[0][0][0] for i in y_pred_list]
        # y_true_list = [i[0] for i in y_true_list]
        print(classification_report(y_true_list, y_pred_list))
        print(confusion_matrix(y_true_list,y_pred_list))
        acc = running_corrects.double().item() / dataset_sizes['test']

    return acc

# Evaluate K top predictions on SIN dataset
def topk_eval_test(model, dataloaders,topk=(1,5,10,20),class_map = mapping_207_reverse):
    model.to(device=device)

    maxk = max(topk)
    
    with torch.no_grad():
        model.eval()

        topk_acc_all = list()

        for i, (inputs, labels) in enumerate(dataloaders['test']):
            batch_size = len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            #get top k prediction acc
            _, preds_topk = outputs.topk(k=maxk, dim=1)

            #top k calculation
            preds_topk = preds_topk.t() 
            target_reshaped = labels.view(1, -1).expand_as(preds_topk)
            correct = (preds_topk == target_reshaped)

            
            list_topk_accs = []  # idx is topk1, topk2, ... etc
            #loop through top k predictions
            for k in topk:
                # get tensor of which topk answer was right
                correct_k  = correct[:k].float().sum(dim=0)
                topk_acc = sum([1 for i in correct_k if i > 0]) / batch_size

                list_topk_accs.append(topk_acc)

            topk_acc_all.append(list_topk_accs)
        topk_acc_all = np.asarray(topk_acc_all).sum(axis=0)/len(topk_acc_all)

    return topk_acc_all

#Shape Bias dataframe and dictionary results
def shape_bias(model, dataloaders,class_map = mapping_207_reverse):
    model.to(device=device)

    #top k initialization
    topk=(1,5,10,20)
    maxk = max(topk)
    
    with torch.no_grad():
        model.eval()

        pred_name = list()
        lab_name = list()
        tex_name = list()
        topk_acc_shape = list()
        topk_acc_texture = list()

        for i, (inputs, labels) in enumerate(dataloaders['test']):
            batch_size = len(labels[0])
            inputs = inputs.to(device)
            shapes = labels[0].to(device)
            textures = labels[1].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            #get top k prediction acc
            _, preds_topk = outputs.topk(k=maxk, dim=1)


            for j in range(inputs.size()[0]):
                #list of pred and label with names
                pred_nameid = class_map[preds[j]]
                pred_name.append(map207_to_16names(pred_nameid))
                
                lab_nameid = class_map[shapes[j]]
                lab_name.append(map207_to_16names(lab_nameid))
                
                tex_nameid = class_map[textures[j]]
                tex_name.append(map207_to_16names(tex_nameid))


                #remapping to 16 numbers #comment out for comparison with 207 class
                # for k in range(5):
                #     preds_topk[j,k] = map207_to_16(class_map[preds_topk[j,k]])
                # #targets
                # shapes[j] = map207_to_16(class_map[shapes[j]])
                # textures[j] = map207_to_16(class_map[textures[j]])

            #top k calculation
            preds_topk = preds_topk.t() 
            target_reshaped = shapes.view(1, -1).expand_as(preds_topk)
            correct = (preds_topk == target_reshaped)
            #for texture
            target_reshaped_texture = textures.view(1, -1).expand_as(preds_topk)
            correct_tex = (preds_topk == target_reshaped_texture)

            
            list_topk_accs = []  # idx is topk1, topk2, ... etc
            list_topk_accs_tex = []
            #loop through top k predictions
            for k in topk:
                # get tensor of which topk answer was right
                correct_k  = correct[:k].float().sum(dim=0)
                topk_acc = sum([1 for i in correct_k if i > 0]) / batch_size

                #for texture
                correct_k_tex  = correct_tex[:k].float().sum(dim=0)
                topk_acc_tex = sum([1 for i in correct_k_tex if i > 0]) / batch_size

                list_topk_accs.append(topk_acc)
                list_topk_accs_tex.append(topk_acc_tex)

            topk_acc_shape.append(list_topk_accs)
            topk_acc_texture.append(list_topk_accs_tex)
        topk_acc_texture = np.asarray(topk_acc_texture).sum(axis=0)/len(topk_acc_texture)
        topk_acc_shape = np.asarray(topk_acc_shape).sum(axis=0)/len(topk_acc_shape)

    #Analysis in dataframe
    shape_bias_df = pd.DataFrame(np.c_[pred_name,lab_name,tex_name],columns=['pred','lab_shape','lab_texture',])
    #define new columns for correct predictions
    shape_bias_df['correct_shape'] = np.where(shape_bias_df['pred'] == shape_bias_df['lab_shape'], True, False)
    shape_bias_df['correct_texture'] = np.where(shape_bias_df['pred'] == shape_bias_df['lab_texture'], True, False)
    
    #calculate number of correct predictions
    correct_pred_both = shape_bias_df[(shape_bias_df['correct_shape']==True) | (shape_bias_df['correct_texture']==True)].count()['pred']
    percent_correct_both = correct_pred_both/len(shape_bias_df)
    correct_pred_s = shape_bias_df[(shape_bias_df['correct_shape']==True)].count()['pred']
    percent_correct_s = correct_pred_s/len(shape_bias_df)
    correct_pred_t = shape_bias_df[(shape_bias_df['correct_texture']==True)].count()['pred']
    percent_correct_t = correct_pred_t/len(shape_bias_df)
    #print(percent_correct)

    # remove those rows where shape = texture, i.e. no cue conflict present
    shape_bias_df_nc = shape_bias_df.loc[shape_bias_df.correct_shape != shape_bias_df.correct_texture]
    #print(shape_bias_df_nc)
    #print(shape_bias_df_nc.pred.unique())

    #shape bias for subclasses
    subclass_result = list()
    for classes in shape_bias_df_nc.pred.unique():
        subclass_df = shape_bias_df_nc[shape_bias_df_nc.pred == classes]
        #print(subclass_df)
        fraction_correct_shape = len(subclass_df.loc[subclass_df.correct_shape==True]) / len(subclass_df)
        fraction_correct_texture = len(subclass_df.loc[subclass_df.correct_texture==True]) / len(subclass_df)
        subclass_result.append(fraction_correct_shape / (fraction_correct_shape + fraction_correct_texture))
        #print(subclass_result)

    subclass_shape_bias_result = dict(zip(shape_bias_df_nc.pred.unique(), subclass_result))
    #print(subclass_shape_bias_result)

    #shape bias for all
    fraction_correct_shape = len(shape_bias_df_nc.loc[shape_bias_df_nc.correct_shape==True]) / len(shape_bias_df_nc)
    fraction_correct_texture = len(shape_bias_df_nc.loc[shape_bias_df_nc.correct_texture==True]) / len(shape_bias_df_nc)
    shape_bias_result = fraction_correct_shape / (fraction_correct_shape + fraction_correct_texture)
    #print(shape_bias_result)
    result_dict = { "percent-correct-both": percent_correct_both,
                    "percent-correct-shape": percent_correct_s,
                    "percent-correct-texture": percent_correct_t,
                    "top-k-acc-shape":topk_acc_shape,
                    "top-k-acc-texture":topk_acc_texture,
                "fraction-correct-shape": fraction_correct_shape,
               "fraction-correct-texture": fraction_correct_texture,
               "subclass-shape-bias": subclass_shape_bias_result,
               "shape-bias": shape_bias_result}


    
    return result_dict, shape_bias_df, shape_bias_df_nc


#Confusion Matrix of 16 SIN class classification
def confusion_matrix_hm(y_pred,y_test, name = 'confusion_matrix',categories = class_16_listed):
    actual = pd.Categorical(y_test, categories=categories)
    predict  = pd.Categorical(y_pred, categories=categories)
    
    confusion_matrix = pd.crosstab(actual, predict, dropna=False, rownames=['Actual'], colnames=['Predicted'])
    
    fig, ax = plt.subplots(figsize=(10,6))  
    sns.heatmap(confusion_matrix, annot=True, fmt='d')
    fig.suptitle(name)
    #plt.show()
    plt.savefig(f'{fig_path}/{name}.png')

#Sample images from Shape Bias prediction
def visualize_model(model, dataloaders, name = 'model_pred', test = True, class_map = mapping_207_reverse, num_images=2):
    model.to(device=device)
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    if test == True:
        select = 'test'
    else:
        select = 'val'

    with torch.no_grad():
        lab_cor = 0
        for i, (inputs, labels) in enumerate(dataloaders[select]):
            if i == 0 : 
                print(type(inputs))
            inputs = inputs.to(device)
            shapes = labels[0].to(device)
            textures = labels[1].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):

                pred_nameid = class_map[preds[j]]
                pred_name = map207_to_16names(pred_nameid)
                lab_nameid = class_map[shapes[j]]
                lab_name = map207_to_16names(lab_nameid)
                tex_nameid = class_map[textures[j]]
                tex_name = map207_to_16names(tex_nameid)

                if (pred_name == lab_name) and (tex_name != lab_name) and (lab_cor ==0):
                    lab_cor = 1
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    nl = '\n'
                    ax.set_title(f'Predicted: {pred_name}, ID: {pred_nameid} {nl} True: {lab_name}, ID: {lab_nameid} {nl} Texture: {tex_name}, ID: {tex_nameid}')
                    imshow(inputs.cpu().data[j])
                    #plt.suptitle('Sample predictions')
                    plt.gcf().set_size_inches(12, 7)
                
                if (pred_name == tex_name) and (tex_name != lab_name):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    nl = '\n'
                    ax.set_title(f'Predicted: {pred_name}, ID: {pred_nameid} {nl} True: {lab_name}, ID: {lab_nameid} {nl} Texture: {tex_name}, ID: {tex_nameid}')
                    imshow(inputs.cpu().data[j])
                    #plt.suptitle('Sample predictions')
                    plt.gcf().set_size_inches(12, 7)

                plt.tight_layout()
                plt.savefig(f'{fig_path}/{name}.png')

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

#plot sampled image
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(f'Sample: {title} ')
    plt.pause(0.1)  # pause a bit so that plots are updated
    
