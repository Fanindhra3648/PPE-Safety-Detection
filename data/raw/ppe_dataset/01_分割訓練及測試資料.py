#注意路徑
#每執行一次就會移動20%到valid
import os
import random
import shutil

# 設定來源資料夾和目標資料夾
train_folder = 'D:/20250731-PPE2286/20250731-ppe2286/train'
val_folder = 'D:/20250731-PPE2286/20250731-ppe2286/valid'

# 確保目標資料夾存在
os.makedirs(val_folder, exist_ok=True)

# 獲取train資料夾中所有的.jpg文件
jpg_files = [f for f in os.listdir(train_folder) if f.endswith('.jpg')]

# 確保每個.jpg文件都有配對的.txt文件
paired_files = [f for f in jpg_files if os.path.exists(os.path.join(train_folder, f.replace('.jpg', '.txt')))]

# 計算20%的檔案數量
num_val_files = int(len(paired_files) * 0.2)

# 隨機選擇20%的檔案
val_files = random.sample(paired_files, num_val_files)

# 移動選擇的檔案到val資料夾
for file in val_files:
    # 移動.jpg檔案
    shutil.move(os.path.join(train_folder, file), os.path.join(val_folder, file))
    # 移動對應的.txt檔案
    shutil.move(os.path.join(train_folder, file.replace('.jpg', '.txt')), os.path.join(val_folder, file.replace('.jpg', '.txt')))

print(f"已隨機選擇並移動 {num_val_files} 組檔案到 {val_folder} 資料夾中")


