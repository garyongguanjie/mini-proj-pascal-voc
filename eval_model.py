import torchvision.models as models
from torch.utils.data import DataLoader
from pascal_functions import *
from PascalVocDataset import PascalVOC
import torch
import torchvision.transforms as transforms
import numpy as np

if __name__ == "__main__":

    ROOT_DIR = './VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    NUM_WORKERS = 0 # Windows cannot set >0
    SAVED_MODEL_NAME = "resnet18-AUG-SGD-LR0.005"

    transformation = transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])

    val_set = PascalVOC(ROOT_DIR,dataset='val',transform=transformation)
    val_set_view = PascalVOC(ROOT_DIR,dataset='val',transform=None)
    val_loader = DataLoader(val_set, batch_size=16, num_workers=NUM_WORKERS)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    model.fc = nn.Linear(512,20)
    model.to(device)
    model.load_state_dict(torch.load(SAVED_MODEL_NAME))
    model.eval()
    # Shows final MAP precision and MAP precision of each class
    true_label,output = model_output(model,val_loader,device)

    # Show pictures with best and worst scores
    for class_name in ['diningtable', 'dog', 'horse','motorbike', 'person']:
        for boolean in [True,False]:
            show_top_n(class_name,5,output,val_set_view,best=boolean)

    max_t = torch.sigmoid(output).numpy().max(axis=0).min()
    interval = (max_t-0.5)/11
    intervals = np.arange(0.5,max_t,interval)
    tailacc(true_label.numpy(),torch.sigmoid(output).numpy(),intervals)