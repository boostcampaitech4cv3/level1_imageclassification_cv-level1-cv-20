import torch
import random
import os
import cv2
import pandas as pd
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np


from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 # 이미지 형 변환
from sklearn.metrics import f1_score, accuracy_score

from transformers import ViTModel, ViTFeatureExtractor
#from efficientnet_pytorch import EfficientNet

import torch.optim as optim
from transformers.optimization import get_cosine_schedule_with_warmup

device = torch.device("cuda:0")

def score_function(real, pred):
    return f1_score(real, pred, average='weighted')

### CONFIG ###

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':30,
    'LEARNING_RATE': 5e-5,
    'BATCH_SIZE':32,
    'SEED':41,
    'MAX_LEN':64,
    'WARMUP_RATIO':0.1,
    'MAX_GRAD_NORM':1,
    'LOG_INTERVAL':200,
    'DROPOUT':0.2,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정


MASK_LABEL = ['incorrect_mask', 'mask1', 'mask2', 'mask3', 'mask4', 'mask5', 'normal']\

train_dir = "./input/data/train"
train_df = pd.read_csv(os.path.join(train_dir, 'train.csv'))


train_image_dir_list = os.listdir(os.path.join(train_dir, 'images'))
train_image_paths = []
valid_image_paths = []
train_label_list = []
valid_label_list = []


n_val = int(len(train_image_dir_list) * 0.2)
n_train = len(train_image_dir_list) - n_val
train_dir_list, val_dir_list = random_split(train_image_dir_list, [n_train, n_val])


for train_image_dir in train_dir_list:
    if train_image_dir.startswith("."):  # "." 로 시작하는 파일은 무시합니다
        continue
    
    gender = train_image_dir.split('_')[1]
    age = int(train_image_dir.split('_')[3])

    if gender =="male":
        gender_label = 0
    else:
        gender_label = 1
    
    if age < 30:
        age_label = 0
    elif age < 60:
        age_label = 1
    else:
        age_label = 2
    

    train_image_path = os.path.join(train_dir, 'images', train_image_dir)

    for file_name in os.listdir(train_image_path):
        if file_name.startswith("."):  # "." 로 시작하는 파일은 무시합니다
            continue
        _file_name, ext = os.path.splitext(file_name)

        train_image_paths.append(os.path.join(train_image_path, file_name))
        if _file_name == "incorrect_mask":
            mask_label = 1
        elif _file_name == "normal":
            mask_label = 2
        else:
            mask_label = 0

        train_label_list.append(mask_label * 6 + gender_label * 3 + age_label)

for valid_image_dir in val_dir_list:
    if valid_image_dir.startswith("."):  # "." 로 시작하는 파일은 무시합니다
        continue
    
    gender = valid_image_dir.split('_')[1]
    age = int(valid_image_dir.split('_')[3])

    if gender =="male":
        gender_label = 0
    else:
        gender_label = 1
    
    if age < 30:
        age_label = 0
    elif age < 60:
        age_label = 1
    else:
        age_label = 2
    

    valid_image_path = os.path.join(train_dir, 'images', valid_image_dir)

    for file_name in os.listdir(valid_image_path):
        if file_name.startswith("."):  # "." 로 시작하는 파일은 무시합니다
            continue
        _file_name, ext = os.path.splitext(file_name)

        valid_image_paths.append(os.path.join(valid_image_path, file_name))
        if _file_name == "incorrect_mask":
            mask_label = 1
        elif _file_name == "normal":
            mask_label = 2
        else:
            mask_label = 0

        valid_label_list.append(mask_label * 6 + gender_label * 3 + age_label)



###  IMAGE AUG  ###

train_transform = A.Compose([
                            A.CenterCrop(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.RandomBrightnessContrast(p=0.5), 
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

test_transform = A.Compose([
                            A.CenterCrop(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])



###  DATASET  ###

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, feature_extractor, transforms, infer=False):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.feature_extractor = feature_extractor
        self.transforms = transforms
        self.infer = infer


    def __getitem__(self, index):
        # Image 읽기
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image'] # transforms(=image augmentation) 적용

        image = self.feature_extractor(image, return_tensors="pt")
        
        # Label
        if self.infer: # infer == True, test_data로부터 label "결과 추출" 시 사용
            return image['pixel_values'][0]
        else: # infer == False
            label = self.label_list[index]
            return image['pixel_values'][0], label

    def __len__(self):
        return len(self.img_path_list)
    

class Custommodel(nn.Module):
    def __init__(self):
        super(Custommodel, self).__init__()
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_classifier = nn.Linear(768, 18)
        self.drop = nn.Dropout(CFG['DROPOUT'])
    
    def forward(self, img):
        cnn_pooler = self.model(pixel_values = img).pooler_output
        out = self.vit_classifier(self.drop(cnn_pooler))

        return out

model = Custommodel()
model.to(device)


# DATALOADER
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
train_dataset = CustomDataset(train_image_paths, train_label_list, feature_extractor, train_transform, infer=False)
trainloader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True)

valid_dataset = CustomDataset(valid_image_paths, valid_label_list, feature_extractor, train_transform, infer=False)
validloader = DataLoader(valid_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False)


###  OPTIMIZER, LOSS, SCHEDULER  ###

optimizer = optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])
criterion = nn.CrossEntropyLoss()

lambda1 = lambda epoch: 0.9 ** CFG['EPOCHS']
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

#t_total = len(trainloader) * CFG['EPOCHS']
#warmup_step = int(t_total * CFG['WARMUP_RATIO'])

#scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

best_score = 0
best_model = None
count = 0

###  TRAINING  ###

for e in range(CFG['EPOCHS']):
    train_loss = []
    model.train()

    for img, label in tqdm((trainloader)):  # mask, gender, age
        optimizer.zero_grad()
        img = img.float().to(device)
        label = label.to(device)

        out = model(img)

        loss = criterion(out, label)
        loss.backward()

        optimizer.step()
        scheduler.step()
        
        train_loss.append(loss.item())

    tr_loss = np.mean(train_loss)
    print(f'Epoch [{e}], Train Loss : [{tr_loss:.5f}]')
    

    ###  VALIDATION  ###

    valid_loss = []
    model_preds = []
    true_labels = []
    model.eval()
    

    with torch.no_grad():
      for img, label in tqdm((validloader)): # mask gender age
        img = img.float().to(device)
        label = label.to(device)

        pred = model(img)

        loss = criterion(pred, label)
        
        valid_loss.append(loss.item())

        model_preds += pred.argmax(1).detach().cpu().numpy().tolist()
        true_labels += label.detach().cpu().numpy().tolist()

    val_loss = np.mean(valid_loss)
    test_weighted_f1 = score_function(true_labels, model_preds)
    acc = accuracy_score(true_labels, model_preds)
    
    if test_weighted_f1 > best_score:
        best_score = test_weighted_f1
        best_model = model
        count = 0
    else:
        count += 1

    if count == 3:
        print('NO LONGER IMPROVED')
        break
        
    print(f'Epoch [{e}], Valid Loss : [{val_loss:.5f}]')
    print(f'Epoch [{e}], Val Score : [{test_weighted_f1:.5f}]')
    print(f'Epoch [{e}], ACC Score : [{acc:.5f}]')



###  MODEL SAVE  ###
#torch.save(model.state_dict(), './model.pth')


test_dir = "./input/data/eval"
test_df = pd.read_csv(os.path.join(test_dir, 'info.csv'))


test_image_dir = os.path.join(test_dir, 'images')
test_image_paths = [os.path.join(test_image_dir, img_id) for img_id in test_df.ImageID]

test_dataset = CustomDataset(test_image_paths, None, feature_extractor, test_transform, infer=True)
testloader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

model_preds = []

with torch.no_grad():
    for batch_id, img in enumerate(testloader):
        img = img.float().to(device)

        out = best_model(img)
        model_preds += out.argmax(1).detach().cpu().numpy().tolist()

test_df['ans'] = model_preds
test_df.to_csv('./submission.csv', index=False)