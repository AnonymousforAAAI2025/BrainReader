import os
import nibabel as nib
import numpy as np

# 数据清理函数
def clean_mgh_folder(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".mgh"):
            file_path = os.path.join(folder_path, file_name)
            img = nib.load(file_path)
            data = img.get_fdata()
            data_cleaned = np.nan_to_num(data)
            img_cleaned = nib.Nifti1Image(data_cleaned, img.affine)
            nib.save(img_cleaned, file_path)  # 覆盖原始文件
            print(f"已处理文件：{file_name}")

def delete_files_without_keyword(folder_path, keyword):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if keyword not in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")


if __name__ == '__main__':
    # 指定数据集文件夹路径
    data_folder = "/media/test/Cui/NSD/data_01_02"
    clean_mgh_folder(data_folder)
    keyword = 'session'
    delete_files_without_keyword(data_folder, keyword)
