import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os

from data.load_data import class_16_listed, map207_to_16names, mapping_207_reverse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

home_path = os.path.expanduser('~')
fig_path = f'{home_path}/scratch/code-snapshots/convolution-vs-attention/src/visualization'

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

def shape_bias(model, dataloaders,class_map = mapping_207_reverse):
    model.to(device=device)
    with torch.no_grad():
        model.eval()

        pred_name = list()
        lab_name = list()
        tex_name = list()

        for i, (inputs, labels) in enumerate(dataloaders['test']):
            
            inputs = inputs.to(device)
            shapes = labels[0].to(device)
            textures = labels[1].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):

                pred_nameid = class_map[preds[j]]
                pred_name.append(map207_to_16names(pred_nameid))
                
                lab_nameid = class_map[shapes[j]]
                lab_name.append(map207_to_16names(lab_nameid))
                
                tex_nameid = class_map[textures[j]]
                tex_name.append(map207_to_16names(tex_nameid))


    #Analysis in dataframe
    shape_bias_df = pd.DataFrame(np.c_[pred_name,lab_name,tex_name],columns=['pred','lab_shape','lab_texture',])
    #define new columns for correct predictions
    shape_bias_df['correct_shape'] = np.where(shape_bias_df['pred'] == shape_bias_df['lab_shape'], True, False)
    shape_bias_df['correct_texture'] = np.where(shape_bias_df['pred'] == shape_bias_df['lab_texture'], True, False)
    
    #calculate number of correct predictions
    correct_pred = shape_bias_df[(shape_bias_df['correct_shape']==True) | (shape_bias_df['correct_texture']==True)].count()['pred']
    percent_correct = correct_pred/len(shape_bias_df)
    #print(percent_correct)

    # remove those rows where shape = texture, i.e. no cue conflict present
    df2 = shape_bias_df.loc[shape_bias_df.correct_shape != shape_bias_df.correct_texture]
   
    fraction_correct_shape = len(df2.loc[df2.correct_shape==True]) / len(shape_bias_df)
    fraction_correct_texture = len(df2.loc[df2.correct_texture==True]) / len(shape_bias_df)
    #print(fraction_correct_shape)
    #print(fraction_correct_texture)
    shape_bias_result = fraction_correct_shape / (fraction_correct_shape + fraction_correct_texture)
    #print(shape_bias_result)
    result_dict = { "percent-correct": percent_correct,
                "fraction-correct-shape": fraction_correct_shape,
               "fraction-correct-texture": fraction_correct_texture,
               "shape-bias": shape_bias_result}


    
    return result_dict, shape_bias_df, df2



def confusion_matrix_hm(y_pred,y_test, name = 'confusion_matrix',categories = class_16_listed):
    actual = pd.Categorical(y_test, categories=categories)
    predict  = pd.Categorical(y_pred, categories=categories)
    
    confusion_matrix = pd.crosstab(actual, predict, dropna=False, rownames=['Actual'], colnames=['Predicted'])
    
    fig, ax = plt.subplots(figsize=(10,6))  
    sns.heatmap(confusion_matrix, annot=True, fmt='d')
    fig.suptitle(name)
    #plt.show()
    plt.savefig(f'{fig_path}/{name}.png')


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
        for i, (inputs, labels) in enumerate(dataloaders[select]):
            if i == 0 : 
                print(type(inputs))
            inputs = inputs.to(device)
            shapes = labels[0].to(device)
            textures = labels[1].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')

                pred_nameid = class_map[preds[j]]
                pred_name = map207_to_16names(pred_nameid)
                
                lab_nameid = class_map[shapes[j]]
                lab_name = map207_to_16names(lab_nameid)
                
                tex_nameid = class_map[textures[j]]
                tex_name = map207_to_16names(tex_nameid)

                nl = '\n'

                ax.set_title(f'Predicted: {pred_name}, ID: {pred_nameid} {nl} True: {lab_name}, ID: {lab_nameid} {nl} Texture: {tex_name}, ID: {tex_nameid}')


                imshow(inputs.cpu().data[j])

                plt.suptitle('Sample predictions')

                plt.gcf().set_size_inches(12, 7)

                plt.savefig(f'{fig_path}/{name}.png')

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)



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
    

# # Get a batch of training data

# #batch size need to be set to 4
# inputs, classes = next(iter(dataloaders['train']))

# print(list(zip(*classes)))
# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)

# imshow(out, title=None)