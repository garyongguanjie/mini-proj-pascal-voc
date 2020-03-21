from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import time
import torch
import torch.nn as nn
from torch.optim import lr_scheduler,Adam,SGD
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score,classification_report
import numpy as np
import pandas as pd
import copy

CLASSES = [ 'aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair',
'cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train','tvmonitor']

def train_model(model, criterion, optimizer, device, dataloaders, scheduler=None, num_epochs=25):
    since = time.time()

    best_precision = 0

    dataset_sizes = {'train': len(dataloaders['train'].dataset),
                     'train_eval': len(dataloaders['train'].dataset),
                     'val': len(dataloaders['val'].dataset)}
    train_precision_list = []; train_loss_list= []; val_precision_list = []; val_loss_list = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        start_epoch_time = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'train_eval', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
  
            # Iterate over data.
            output_ls = []
            label_ls = [] 
            
            for data in dataloaders[phase]:
                inputs = data[0]
                labels_cpu = data[1]

                inputs = inputs.to(device)
                labels = labels_cpu.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
               
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    if phase == 'train_eval' or phase == 'val':
                        output_ls.append(outputs.cpu().detach())
                        label_ls.append(labels_cpu)
                        
    
                # statistics
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'train_eval' or phase == 'val':
                epoch_output = torch.cat(output_ls)
                epoch_label = torch.cat(label_ls)
                epoch_precision = average_precision_score(epoch_label,epoch_output,average='macro')
                print('{} Loss: {:.4f} Precision: {:.4f}'.format(phase, epoch_loss, epoch_precision))
            
            if phase == 'train' and scheduler != None:
                scheduler.step()

            if phase == "train_eval":
                train_precision_list.append(epoch_precision)
                train_loss_list.append(epoch_loss)
                
            elif phase == "val":
                val_precision_list.append(epoch_precision)
                val_loss_list.append(epoch_loss)
                
            # deep copy the model
            if phase == 'val' and epoch_precision > best_precision:
                best_precision = epoch_precision
                best_model_wts = copy.deepcopy(model.state_dict())
        end_epoch_time = time.time()
        print("{:.0f}s".format(end_epoch_time - start_epoch_time))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Precision: {:4f}'.format(best_precision))

    # load best model weights
    model.load_state_dict(best_model_wts)
    plots = (train_precision_list,train_loss_list,val_precision_list,val_loss_list)
    return model, plots

def plot_model_metrics(plots,name):
    train_precision_list,train_loss_list,val_precision_list,val_loss_list = plots
    plot(train_precision_list,val_precision_list,"precision",name)
    plot(train_loss_list,val_loss_list,"loss",name)


def plot(train,val,metric,name):
    plt.title(name)
    plt.plot(train,label="train {}".format(metric))
    plt.plot(val,label="val {}".format(metric))
    plt.legend(loc="best")
    plt.savefig("{}-{}".format(name,metric))
    plt.close()

def model_output(model,dataloader,device):
    output_ls = []
    label_ls = []
    with torch.no_grad():
        for data in dataloader:
            inputs = data[0]
            labels_cpu = data[1]
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            output_ls.append(outputs.cpu().detach())
            label_ls.append(labels_cpu)
    final_output = torch.cat(output_ls)
    true_label = torch.cat(label_ls)
    df = pd.DataFrame(columns=['class','MAP'])

    for i,class_name in enumerate(CLASSES):
        mean_ap_score = average_precision_score(true_label[:,i],final_output[:,i],average='macro')
        print("Mean Average precision of {}: {:.4f}".format(class_name,mean_ap_score))
        df.loc[i] = [class_name,round(mean_ap_score,3)]
        
    final_precision = average_precision_score(true_label,final_output,average='macro')
    df.loc[20] = ["macro avg all",round(final_precision,3)]
    df.to_csv("MAPscores.csv",index=False)
    print("Final mean average precision on val set {:4f}".format(final_precision))
    return true_label,final_output

def show_top_n(class_name,n,output,dataset,best=True):

    index = CLASSES.index(class_name)
    if best:
        final_images = output[:,index].numpy().argsort()[::-1][:n]
    else:
        final_images = output[:,index].numpy().argsort()[:n]
    
    num_rows = max(1,n/5 if n%5==0 else n/5+1)
    f, axarr = plt.subplots(int(num_rows),5,figsize=(10,5))
    if best:
        name = "best"
    else:
        name = "worst"
    f.suptitle('{}-{}-{}'.format(class_name,name,n))
    for i,e in enumerate(final_images): 
        col = i % 5
        row  = i // 5
        image = dataset[e][0].numpy().transpose(1,2,0)
        if num_rows >1:
            axarr[row,col].imshow(image)
        else:
            axarr[col].imshow(image)

    plt.savefig("{}-{}-{}".format(class_name,name,n))
    plt.close()

def tailacc(y_true,y_pred,intervals):
    
    cols_list = copy.copy(CLASSES)
    cols_list.insert(0,"t-values")
    cols_list.append("average")
    df = pd.DataFrame(columns =cols_list)

    tail_accuracies = []
    t_value = []
    for index,t in enumerate(intervals):
        tail_acc_t = 0 #tail accuracy for each t with macro weight 
        interval_tail_acc = [t]
        
        for i in range(20):
            pred_indicator = (y_pred>t)
            answer = 1/pred_indicator.sum() * (y_true*pred_indicator).sum()
            interval_tail_acc.append(round(answer,2))
            tail_acc_t += answer
        avg = tail_acc_t/20
        interval_tail_acc.append(round(avg,2))
        df.loc[index]= interval_tail_acc
        tail_accuracies.append(avg)
        t_value.append(t)
    df.to_csv("tail-accuracies-t-values.csv",index=False)
    plt.title("tail accuracies vs t-values")
    plt.plot(t_value,tail_accuracies)
    plt.savefig('tail_acc_vs_t_values')
    plt.close()