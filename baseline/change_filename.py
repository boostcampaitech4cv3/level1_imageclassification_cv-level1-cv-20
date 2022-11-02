import os
import shutil

img_path = ["./face_input/train/images"]
new_path = ["./data/train/images"]

_file_names = {
        "mask1", "mask2", "mask3", "mask4", "mask5", "incorrect_mask", "normal",
        "mask11", "mask21", "mask31", "mask41", "mask51", "incorrect_mask1", "normal1",
        # "mask12", "mask22", "mask32", "mask42", "mask52", "incorrect_mask2", "normal2"
}

for idx, path in enumerate(img_path):
    profiles = os.listdir(path) # path 안에 있는 파일, 폴더를 가져옴 #ID_gender_race_age
    for profile in profiles:
        name = os.path.basename(profile)
        age = name[-2:]
        # print(age)
        if profile.startswith(".") or int(age) < 55:  # "." 로 시작하는 파일은 무시합니다
            continue
        # print(age)
        img_folder = os.path.join(path, profile) # paht: 폴더 경로 + profile + (파일 or 폴더)
        for file_name in os.listdir(img_folder):
            # print(file_name)
            _file_name, ext = os.path.splitext(file_name)
            if _file_name not in _file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                continue
            # new = _file_name[:-1]+ ext # 1 빼기
            new = _file_name + "1" + ext # 1 더하기
            print(_file_name, new)
            exist_img_path = os.path.join(img_folder, file_name) # /opt/ml/face_input/train/images/profile/file_name
            new_file_path = os.path.join(img_folder, new) # /opt/ml/face_input_train/images/profile/new
            new_img_path = os.path.join(new_path[idx], profile, new)  # /opt/ml/data/train/images/profile
            os.rename(exist_img_path, new_file_path)
            shutil.copy2(new_file_path ,new_img_path)