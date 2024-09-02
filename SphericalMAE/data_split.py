import os
import nibabel as nib
import numpy as np

#训练数据预处理
def Find_mgh(root_dir):
    mgh_paths = []
    # 遍历根目录下的所有子文件夹
    for root, dirs, files in os.walk(root_dir):
        # 查找名为 'fsaverage' 的子文件夹
        if 'fsaverage' in dirs:
            fsaverage_dir = os.path.join(root, 'fsaverage')

            # 查找名为 'betas_fithrf' 的子文件夹
            betas_fithrf_dirs = [d for d in os.listdir(fsaverage_dir) if
                                 os.path.isdir(os.path.join(fsaverage_dir, d)) and d == 'betas_fithrf']

            # 遍历 'betas_fithrf' 子文件夹
            for betas_fithrf_dir in betas_fithrf_dirs:
                betas_fithrf_path = os.path.join(fsaverage_dir, betas_fithrf_dir)

                # 查找包含 'session' 的 .mgh 文件
                session_mgh_files = [f for f in os.listdir(betas_fithrf_path) if os.path.isfile(
                    os.path.join(betas_fithrf_path, f))  and'session' in f and f.endswith('.mgh')]

                # 构建完整路径并添加到列表
                mgh_paths.extend(
                    [os.path.join(betas_fithrf_path, session_mgh_file) for session_mgh_file in session_mgh_files])
    return mgh_paths

def Find_mgh2(root_dir):
    mgh_paths = []
    # 遍历根目录下的所有子文件夹
    for root, dirs, files in os.walk(root_dir):
        # 查找名为 'fsaverage' 的子文件夹
        if 'fsaverage' in dirs:
            fsaverage_dir = os.path.join(root, 'fsaverage')

            # 查找名为 'betas_fithrf' 的子文件夹
            betas_fithrf_dirs = [d for d in os.listdir(fsaverage_dir) if
                                 os.path.isdir(os.path.join(fsaverage_dir, d)) and d == 'betas_fithrf']

            # 遍历 'betas_fithrf' 子文件夹
            for betas_fithrf_dir in betas_fithrf_dirs:
                betas_fithrf_path = os.path.join(fsaverage_dir, betas_fithrf_dir)

                # 查找包含 'session' 的 .mgh 文件
                session_mgh_files = [f for f in os.listdir(betas_fithrf_path) if os.path.isfile(
                    os.path.join(betas_fithrf_path, f)) and 'rh' in f and'session' in f and f.endswith('.mgh')]

                # 构建完整路径并添加到列表
                mgh_paths.extend(
                    [os.path.join(betas_fithrf_path, session_mgh_file) for session_mgh_file in session_mgh_files])
    return mgh_paths
def slice_and_save_mgh_files(mgh_paths, output_dir='data_split'):
    """
    Args:
        mgh_paths (list): 包含.mgh文件路径的列表。
        output_dir (str): 输出目录，存放切片后的.mgh文件，默认为'data_split'。
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    for idx, mgh_path in enumerate(mgh_paths):
        # 使用Nibabel读取.mgh文件
        img = nib.load(mgh_path)
        data = img.get_fdata()

        # 沿着第四个维度切片
        slices = np.split(data, data.shape[-1], axis=-1)

        # 存储切片为新的.mgh文件
        for i, slice_data in enumerate(slices):
            # 保持前三个维度不变
            slice_data_3d = np.squeeze(slice_data, axis=-1).astype(np.float32)
            slice_data_3d = slice_data_3d.reshape(163842,-1)
            slice_img = nib.MGHImage(slice_data_3d, affine=img.affine)
            slice_filename = f"slice_{idx + 1}_{i + 1}.mgh"
            slice_filepath = os.path.join(output_dir, slice_filename)
            nib.save(slice_img, slice_filepath)
            #nib.mghformat.write(slice_data_3d, slice_filename)
if __name__ == '__main__':
    input_directory1 = 'D:/NSD/nsddata_betas/ppdata'
    
    output_directory = '/mnt/mydrive/NSD/data_split_04_05'
    # output_directory2 = 'C:/Users/12993/Desktop/Spherical-MAE-main/test_split'
    # #test_path = 'D:/NSD/data_split/slice_1_1.mgh'
    # #data = nib.load(test_path)
    # #data =data.get_fdata()
    # #print(data.shape)
    mgh_path1=Find_mgh('/mnt/mydrive/NSD/nsddata_betas/ppdata/subj04')
    mgh_path2=Find_mgh('/mnt/mydrive/NSD/nsddata_betas/ppdata/subj05')
    mgh_path = mgh_path1 + mgh_path2
    # #print(len(mgh_path))
    # #slice_and_save_mgh_files(mgh_path, output_directory)
    # #print(len(slice_path))
    # mgh_path = ['C:/Users/12993/Desktop/test.mgh']
    slice_and_save_mgh_files(mgh_path, output_directory)
   
