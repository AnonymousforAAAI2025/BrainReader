import os
import nibabel as nib
import numpy as np

def read_mgh_files(root1, root2, output_dir):
    lh_files = {}
    rh_files = {}

    # 遍历第一个路径下的所有文件
    for filename in os.listdir(root1):
        if filename.endswith(".mgh"):
            if "lh." in filename:
                session_name = filename.split("_")[-1]  # 获取session名
                session_name = session_name.split(".")[-2]
                if session_name not in lh_files:
                    lh_files[session_name] = []
                lh_files[session_name].append(os.path.join(root1, filename))
            elif "rh." in filename:
                session_name = filename.split("_")[-1]  # 获取session名
                session_name = session_name.split(".")[-2]
                if session_name not in rh_files:
                    rh_files[session_name] = []
                rh_files[session_name].append(os.path.join(root1, filename))

    # 遍历第二个路径下的所有文件
    for filename in os.listdir(root2):
        if filename.endswith(".mgh"):
            if "lh." in filename:
                session_name = filename.split("_")[-1]  # 获取session名
                session_name = session_name.split(".")[-2]
                if session_name not in lh_files:
                    lh_files[session_name] = []
                lh_files[session_name].append(os.path.join(root2, filename))
            elif "rh." in filename:
                session_name = filename.split("_")[-1]  # 获取session名
                session_name = session_name.split(".")[-2]
                if session_name not in rh_files:
                    rh_files[session_name] = []
                rh_files[session_name].append(os.path.join(root2, filename))

    for session, lh_list in lh_files.items():
        if session in rh_files and len(lh_list) == len(rh_files[session]):
            for i in range(len(lh_list)):
                lh_data = nib.load(lh_list[i]).get_fdata()
                rh_data = nib.load(rh_files[session][i]).get_fdata()
                # 沿着第四个维度切片
                slices_l = np.split(lh_data, lh_data.shape[-1], axis=-1)
                slices_r = np.split(rh_data, rh_data.shape[-1], axis=-1)

                # 存储切片为新的.mgh文件
                for j, slice_l in enumerate(slices_l):
                    slice_l = slice_l.reshape(-1, 1).astype(np.float32)
                    #slice_l = slice_l[:163842,:]
                    slice_r = slices_r[j].reshape(-1, 1).astype(np.float32)
                    #slice_r = slice_r[:163842, :]
                    slice = np.concatenate([slice_l, slice_r], axis=0)
                    #print(slice.shape)
                    slice_img = nib.MGHImage(slice, affine=np.eye(4))
                    subj_prefix = "subj01" if lh_list[i].startswith(root1) else "subj02"
                    folder_name = os.path.basename(os.path.dirname(lh_list[i]))
                    slice_filename = f"{subj_prefix}_{folder_name}_{session}_{j+1}.mgh"
                    slice_filepath = os.path.join(output_dir, slice_filename)
                    nib.save(slice_img, slice_filepath)

if __name__ == '__main__':
    # 调用函数并传入包含.mgh文件的目录路径
    root1 = "/media/test/Cui/NSD/nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf_GLMdenoise_RR"
    root2 = "/media/test/Cui/NSD/nsddata_betas/ppdata/subj02/fsaverage/betas_fithrf_GLMdenoise_RR"
    output_dir = '/media/test/Cui/NSD/data_01_02'
    read_mgh_files(root1, root2, output_dir)
