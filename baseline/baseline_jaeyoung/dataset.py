import os

import cv2
import pandas as pd
from hyperparameter import HyperParameter
from torch.utils.data import Dataset, random_split


class MaskLabels():
    MASK = 0
    INCORRECT = 1
    NORMAL = 2

    @classmethod
    def from_str(self, value: str) -> int:
        value = value.lower()
        if value in ["mask1", "mask2", "mask3", "mask4", "mask5"]:
            return self.MASK
        elif value == "incorrect_mask":
            return self.INCORRECT
        else:
            return self.NORMAL


class GenderLabels():
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(self, value: str) -> int:
        value = value.lower()
        if value == "male":
            return self.MALE
        elif value == "female":
            return self.FEMALE
        else:
            raise ValueError(
                f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels():
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(self, value: int) -> int:
        if value < 30:
            return self.YOUNG
        elif value < 60:
            return self.MIDDLE
        else:
            return self.OLD


class AgeLabelsAdjusted():
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(self, value: int) -> int:
        if value < 30:
            return self.YOUNG
        elif value < 57:
            return self.MIDDLE
        else:
            return self.OLD



class SplitByHumanDataset(Dataset):
    """
        사람을 기준으로 dataset을 나눴음. Test score과 Validation Score간의 격차를 줄일 수 있다
    
    """
    def _get_images(self):
        IMG_EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG",
                          ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP"]
        MASK_TYPES = ["mask1", "mask2", "mask3",
                      "mask4", "mask5", "incorrect_mask", "normal"]
        df = pd.read_csv(HyperParameter.TRAIN_CSV_DIR)
        for i in self.df_index:
            row = df.iloc[i]
            p = os.path.join(HyperParameter.TRAIN_IMAGE_DIR, row.path)
            files = os.listdir(p)
            for mask_type in MASK_TYPES:
                for ext in IMG_EXTENSIONS:
                    if mask_type+ext in files:
                        self.image_paths.append(os.path.join(p, mask_type+ext))
                        mask=MaskLabels.from_str(mask_type)
                        gender=GenderLabels.from_str(row.gender)
                        age=AgeLabelsAdjusted.from_number(row.age.item()) if self.train else AgeLabels.from_number(row.age.item())
                        self.mask_labels.append(mask)
                        self.gender_labels.append(gender)
                        self.age_labels.append(age)
                        self.classes.append(SplitByHumanDataset.multi_to_single(mask,gender,age))


    def __init__(self, df_index: list ,transform=None,train=True) -> None:
        self.image_paths = []
        self.mask_labels = []
        self.gender_labels = []
        self.age_labels = []
        self.classes=[]
        self.transform = transform
        self.df_index=df_index
        self.train=train
        self._get_images()


    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label=[self.mask_labels[index], self.gender_labels[index], self.age_labels[index]]
        if self.transform:
            image = self.transform(image=image)["image"]
        return image,label

    def __len__(self):
        return len(self.image_paths)
    
    @staticmethod
    def split_train_val():
        df=pd.read_csv(HyperParameter.TRAIN_CSV_DIR)
        val=int(len(df)*HyperParameter.VALIDATION_RATIO)
        train=len(df)-val
        t,v=random_split(list(range(len(df))),[train,val])
        return t,v


    
    @staticmethod
    def multi_to_single(mask_label:int,gender_label:int,age_label:int) -> int:
        """
            mask, gender, age로 이루어진 label들을 하나의 정수로 변환
        """
        return mask_label*6+gender_label*3+age_label
    
    @staticmethod
    def single_to_multi(label:int):
        """
            mask, gender, age로 이루어진 label들을 하나의 정수로 변환
        """
        mask= label//6
        gender=(label//3)%2
        age=label%3
        return mask,gender,age

class SplitByHumanWithFaceDataset(Dataset):
    """
        사람을 기준으로 dataset을 나눴음. Test score과 Validation Score간의 격차를 줄일 수 있다
    
    """
    def _get_images(self):
        IMG_EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG",
                          ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP"]
        MASK_TYPES = ["mask1", "mask2", "mask3",
                      "mask4", "mask5", "incorrect_mask", "normal"]
        df = pd.read_csv(HyperParameter.TRAIN_CSV_DIR)
        df_face= pd.read_csv(HyperParameter.TRAIN_FACE_CSV_DIR)
        for i in self.df_index:
            row = df.iloc[i]
            p = os.path.join(HyperParameter.TRAIN_IMAGE_DIR, row.path)
            files = os.listdir(p)
            for mask_type in MASK_TYPES:
                for ext in IMG_EXTENSIONS:
                    if mask_type+ext in files:
                        self.image_paths.append(os.path.join(p, mask_type+ext))
                        mask=MaskLabels.from_str(mask_type)
                        gender=GenderLabels.from_str(row.gender)
                        age=AgeLabelsAdjusted.from_number(row.age.item()) if self.train else AgeLabels.from_number(row.age.item())
                        self.mask_labels.append(mask)
                        self.gender_labels.append(gender)
                        self.age_labels.append(age)
                        self.classes.append(SplitByHumanWithFaceDataset.multi_to_single(mask,gender,age))
                        
                        
            if self.train:
                row = df_face.iloc[i]
                p = os.path.join(HyperParameter.TRAIN_FACE_IMAGE_DIR, row.path)
                files = os.listdir(p)
                for mask_type in MASK_TYPES:
                    for ext in IMG_EXTENSIONS:
                        if mask_type+ext in files:
                            self.image_paths.append(os.path.join(p, mask_type+ext))
                            mask=MaskLabels.from_str(mask_type)
                            gender=GenderLabels.from_str(row.gender)
                            age=AgeLabelsAdjusted.from_number(row.age.item()) if self.train else AgeLabels.from_number(row.age.item())
                            self.mask_labels.append(mask)
                            self.gender_labels.append(gender)
                            self.age_labels.append(age)
                            self.classes.append(SplitByHumanWithFaceDataset.multi_to_single(mask,gender,age))


    def __init__(self, df_index: list ,transform=None,train=True) -> None:
        self.image_paths = []
        self.mask_labels = []
        self.gender_labels = []
        self.age_labels = []
        self.classes=[]
        self.transform = transform
        self.df_index=df_index
        self.train=train
        self._get_images()


    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label=[self.mask_labels[index], self.gender_labels[index], self.age_labels[index]]
        if self.transform:
            image = self.transform(image=image)["image"]
        return image,label

    def __len__(self):
        return len(self.image_paths)
    
    @staticmethod
    def split_train_val():
        df=pd.read_csv(HyperParameter.TRAIN_CSV_DIR)
        val=int(len(df)*HyperParameter.VALIDATION_RATIO)
        train=len(df)-val
        t,v=random_split(list(range(len(df))),[train,val])
        return t,v

    
    @staticmethod
    def multi_to_single(mask_label:int,gender_label:int,age_label:int) -> int:
        """
            mask, gender, age로 이루어진 label들을 하나의 정수로 변환
        """
        return mask_label*6+gender_label*3+age_label
    
    @staticmethod
    def single_to_multi(label:int):
        """
            mask, gender, age로 이루어진 label들을 하나의 정수로 변환
        """
        mask= label//6
        gender=(label//3)%2
        age=label%3
        return mask,gender,age


class TestDataset(Dataset):


    def __init__(self,df, transform):
        self.transform=transform
        self.image_paths=[os.path.join(HyperParameter.TEST_IMAGE_DIR, img_id) for img_id in df.ImageID]

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]
        return image

    def __len__(self):
        return len(self.image_paths)

