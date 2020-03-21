from pascal_functions import *
from PascalVocDataset import PascalVOC
import torchvision.models as models
from torch.optim import lr_scheduler,Adam,SGD
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms

if __name__ == "__main__":
    ROOT_DIR = './VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    NUM_EPOCHS = 25
    NUM_WORKERS = 0 #Windows cannot set more than 1 workers for dataloader
    LR = 0.005
    SAVED_MODEL_NAME = "resnet18-AUG-SGD-LR0.005"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512,20)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    train_transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.RandomErasing(p=0.5,scale=(0.10, 0.15),ratio=(1, 1)),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])

    transformation = transforms.Compose([
                                    transforms.Resize((224,224)),#I dont want to crop off a dog's head
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])

    
    train_set = PascalVOC(ROOT_DIR,dataset='train',transform=train_transform)
    train_eval_set = PascalVOC(ROOT_DIR,dataset='train',transform=transformation)
    val_set = PascalVOC(ROOT_DIR,dataset='val',transform=transformation)
    val_set_view = PascalVOC(ROOT_DIR,dataset='val',transform=None)

    train_loader = DataLoader(train_set, batch_size=16, num_workers=NUM_WORKERS,shuffle=True)
    train_eval_loader = DataLoader(train_eval_set,batch_size=16,num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=16, num_workers=NUM_WORKERS)
    
    dataloaders = {'train': train_loader,'train_eval':train_loader,'val': val_loader}

    optimizer = SGD(list(model.parameters()),lr=LR, momentum=0.9, weight_decay=0)
    final_model, plots = train_model(model,criterion,optimizer,device,dataloaders,num_epochs=NUM_EPOCHS)

    plot_model_metrics(plots,"pascal_voc-AUG")

    torch.save(final_model.state_dict(),SAVED_MODEL_NAME)