import os, math, glob, itertools
from typing import Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import interpn
import pydicom
import torch


def load_dicom(data_path: str, debug: bool = False, column_name: str = "DICOMPath", directory_levels: int = 2, save_voxel_array: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dicom file
    Args:
        data_path (str): path to dicom file
        column_name (str): column name of dicom path
        directory_levels (int): number of directory levels
        save_voxel_array (bool): whether to save voxel array
    Returns:
        voxel_arr (np.ndarray): voxel array. Shape is (N,W,H,D).
        spacing (np.ndarray): spacing. Shape is (N,3).
        origin (np.ndarray): origin. Shape is (N,3).
        path_arr (np.ndarray): path list. Shape is (N,).
    """
    if os.path.isdir(data_path):
        voxel_arr, spacing, origin = load_dicom_as_voxels(data_path)
        voxel_arr, spacing, origin = [arr[None] for arr in [voxel_arr, spacing, origin]]
        path_arr = np.array([data_path])
    elif os.path.isfile(data_path):
        extension = os.path.splitext(data_path)[1]
        if extension == ".csv":
            df = pd.read_csv(data_path)
            path_arr = df[column_name].values
        elif extension == ".xlsx":
            df = pd.read_excel(data_path)
            path_arr = df[column_name].values
        else:
            raise ValueError(f"{extension} is not supported")

        if path_arr.dtype in [np.int64, np.int32, np.int16, np.int8, np.float128, np.float64, np.float32, np.float16]:
            path_arr = path_arr.astype(int)
            path_arr = [f"{p:08d}" for p in path_arr]

        if debug:
            path_arr = path_arr[:2]

        dirname = os.path.dirname(data_path)
        path_list = []
        for p in path_arr:        
            while len(p.split("/")) < directory_levels:
                entries = os.listdir(os.path.join(dirname, p))
                directories = [os.path.join(p, entry) for entry in entries if os.path.isdir(os.path.join(dirname, p, entry))]
                if len(directories) == 1:
                    p = directories[0]
                elif len(directories) == 0:
                    raise ValueError(f"{dirname}/{p} has no directory")
                else:
                    raise ValueError(f"{dirname}/{p} has multiple directories: {directories}")
            path_list.append(os.path.join(dirname, p))
        path_arr = np.array(path_list)

        voxel_arr, spacing, origin = [], [], []
        for p in path_arr:
            v, s, o = load_dicom_as_voxels(p, save_voxel_array)
            voxel_arr.append(v)
            spacing.append(s)
            origin.append(o)
        voxel_arr, spacing, origin = [np.array(arr) for arr in [voxel_arr, spacing, origin]]
    else:
        raise ValueError(f"{data_path} is not a file or directory")
    return voxel_arr, spacing, origin, path_arr


def load_dicom_as_voxels(dirname: str, save: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load dicom file as voxels
    Args:
        dirname (str): path to dicom file
        save (bool): whether to save voxel array
    
    Returns:
        voxel_arr (np.ndarray): voxel array. Shape is (W,H,D).
        spacing (np.ndarray): spacing. Shape is (3,).
        origin (np.ndarray): origin. Shape is (3,).
    """
    assert os.path.exists(dirname)
    
    number_of_dcm_files = len(glob.glob(os.path.join(dirname, "*.DCM")))
    print(f"{dirname} has {number_of_dcm_files} DCM files")
    
    if number_of_dcm_files > 0:
        if os.path.exists(os.path.join(dirname, "voxel_array.npy")):
            voxel_array = np.load(os.path.join(dirname, "voxel_array.npy"))
            dcm_file_path_list = [os.path.join(dirname, "{:08d}.DCM".format(1))]
        else:
            dcm_file_path_list = [os.path.join(dirname, f"{i:08d}.DCM") for i in range(1, number_of_dcm_files+1)]
            voxel_array = np.zeros([dcm_file.Rows, dcm_file.Columns, len(dcm_file_path_list)])
            for i, path in enumerate(dcm_file_path_list):
                print(f"\rcomputing in {i}/{len(dcm_file_path_list)}", end="")
                dcm_file = pydicom.dcmread(path)
                voxel_array[:,:,i] = dcm_file.pixel_array

            if save:
                np.save(os.path.join(dirname, "voxel_array.npy"), voxel_array)
        
        dcm_file = pydicom.dcmread(dcm_file_path_list[0])
        spacing = np.array(list(dcm_file.PixelSpacing) + [-dcm_file.SliceThickness])
        origin = np.array(dcm_file.ImagePositionPatient)
    else:
        raise ValueError(f"{dirname} has no DCM files")

    return voxel_array.swapaxes(0,1).astype("float32"), spacing, origin


def load_data(csv_path: str, downsampling: int = 4, handling_samples_over_max_length: str = "del", max_z_length: int = 320, target_spacing: float = 0.5, debug: bool = False):
    """Load voxel array and landmarks from csv file.
    Args:
        csv_path (str): Path to csv file.
        downsampling (int): Downsampling rate.
        handling_samples_over_max_length (str): How to handle samples whose z length is over max_z_length. "del" or "cut".
        max_z_length (int): Max z length.
        target_spacing (float): Target spacing.
        debug (bool): If True, load only 2 samples.
    """
    voxel_arr, landmarks, spacing, origin, path_list = get_voxel_array(csv_path, downsampling, handling_samples_over_max_length, max_z_length, debug) #(N,W,H,D), (N,3,3), (N,3), (N,3)
    if target_spacing is not None:
        voxel_arr, spacing, origin = adjust_spacing_of_voxel_array(voxel_arr, spacing, origin, target_spacing, downsampling) #(N,W,H,D), (N,3), (N,3)
    landmarks = transform_landmarks_to_downsampled_voxel(landmarks, spacing, origin, downsampling) #(N,3,3)
    return voxel_arr[:,None], landmarks, spacing, origin, path_list


def get_voxel_array(csv_path: str, downsampling: int = 4, handling_samples_over_max_length: str = "del", max_z_length: int = 320, debug: bool = False):
    """Load voxel array and landmarks from csv file.
    Args:
        csv_path (str): Path to csv file.
        downsampling (int): Downsampling rate.
        handling_samples_over_max_length (str): How to handle samples whose z length is over max_z_length. "del" or "cut".
        max_z_length (int): Max z length.
        debug (bool): If True, load only 5 samples.
    Returns:
        voxel_arr (np.ndarray): Voxel array. Shape is (N,W,H,D).
        landmarks (np.ndarray): Landmarks. Shape is (N,3,3).
        spacing (np.ndarray): Spacing. Shape is (N,3).
        origin (np.ndarray): Origin. Shape is (N,3).
        path_list (list): List of path to voxel array.
    """

    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["RCC_x", "RCC_y", "RCC_z", "LCC_x", "LCC_y", "LCC_z", "NCC_x", "NCC_y", "NCC_z"])
    dirname = os.path.dirname(csv_path)

    if debug:
        df = df.dropna(subset=["CTquality"])
        df = df.iloc[:6]
    path_list = df["DICOMPath"].values
    obs, indices = [], []
    for i, p in enumerate(path_list):
        arr = np.load(os.path.join(dirname, p, "voxel_array.npy")).astype("float32").swapaxes(0,1)
        if handling_samples_over_max_length == "del":
            if arr.shape[2] == max_z_length:
                obs.append(arr[::downsampling,::downsampling,::downsampling])
                indices.append(i)
            else:
                print("Sample {} is deleted because its length {} is not equal to {}.".format(p, arr.shape[2], max_z_length))
        elif handling_samples_over_max_length == "cut":
            if arr.shape[2] >= max_z_length:
                obs.append(arr[::downsampling,::downsampling,:max_z_length:downsampling])
                indices.append(i)
            else:
                print("Sample {} is deleted because its length {} is lower than {}.".format(p, arr.shape[2], max_z_length))
    path_list = [path_list[i] for i in indices]
    landmarks = df[["RCC_x", "RCC_y", "RCC_z", "LCC_x", "LCC_y", "LCC_z", "NCC_x", "NCC_y", "NCC_z"]].values[indices].astype("float32").reshape(len(obs), 3, 3) #(N,3,3)
    spacing = df[["spacing_x", "spacing_y", "spacing_z"]].values[indices].astype("float32") # signed spacing (N,3)
    origin = df[["origin_x", "origin_y", "origin_z"]].values[indices].astype("float32") #(N,3)
    return np.stack(obs), landmarks, spacing, origin, np.array(path_list)


def get_global_voxel_array(voxel_arr: np.ndarray, spacing: np.ndarray, origin: np.ndarray, target_spacing: Optional[float] = None, downsampling: int = 1):
    """Convert voxel array to global voxel array.
    Args:
        voxel_arr (np.ndarray): Voxel array. Shape is (N,W,H,D).
        spacing (np.ndarray): Spacing. Shape is (N,3).
        origin (np.ndarray): Origin. Shape is (N,3).
        target_spacing (Optional[float]): Target spacing.
        downsampling (int): Downsampling rate.
    
    Returns:
        g_voxel_arr (np.ndarray): Global voxel array. Shape is (N,1,W,H,D).
        g_spacing (np.ndarray): Global spacing. Shape is (N,3).
        g_origin (np.ndarray): Global origin. Shape is (N,3).
    """
    if target_spacing is not None:
        g_voxel_arr, g_spacing, g_origin = adjust_spacing_of_voxel_array(voxel_arr, spacing, origin, target_spacing, downsampling) # (N,W,H,D), (N,3), (N,3)
    else:
        g_voxel_arr, g_spacing, g_origin = voxel_arr, spacing, origin
    g_voxel_arr = g_voxel_arr[:,None,::downsampling,::downsampling,::downsampling] # (N,1,W,H,D)
    return g_voxel_arr, g_spacing, g_origin


def adjust_spacing_of_voxel_array(voxel_arr: np.ndarray, spacing: np.ndarray, origin: np.ndarray, target_spacing: Union[float, List, np.ndarray] = 0.35, downsampling: int = 1):
    """Adjust spacing of voxel array.
    Args:
        voxel_arr (np.ndarray): Voxel array. Shape is (N,W,H,D).
        spacing (np.ndarray): Spacing. Shape is (N,3).
        origin (np.ndarray): Origin. Shape is (N,3).
        target_spacing (float or list or np.ndarray): Target spacing.
    Returns:
        voxel_arr (np.ndarray): Voxel array. Shape is (N,W,H,D).
        spacing (np.ndarray): Spacing. Shape is (N,3).
        origin (np.ndarray): Origin. Shape is (N,3).
    """
    assert voxel_arr.ndim == 4
    assert spacing.ndim == 2
    assert voxel_arr.shape[0] == spacing.shape[0]

    if isinstance(target_spacing, float):
        target_spacing = target_spacing * np.ones(3)
    elif isinstance(target_spacing, list):
        target_spacing = np.array(target_spacing)
    spacing = spacing * downsampling # signed downsampled spacing
    target_spacing = target_spacing * downsampling # unsigned downsampled target spacing

    new_voxel_arr = np.zeros_like(voxel_arr) #(N,W,H,D)
    for i in range(voxel_arr.shape[0]):
        grid_arr = tuple([spacing[i,j]*np.arange(voxel_arr.shape[j+1]) for j in range(3)]) #(3,W/H/D)
        new_grid_arr = np.stack(np.meshgrid(*[target_spacing[j]*np.arange(voxel_arr.shape[j+1]) + 0.5 * (spacing[i,j] - target_spacing[j]) * voxel_arr.shape[j+1] for j in range(3)], indexing="ij"), axis=-1) #(W,H,D,3)
        new_voxel_arr[i] = interpn(grid_arr, voxel_arr[i], new_grid_arr, method="linear", bounds_error=False, fill_value=0.0) #(W,H,D)

    origin = origin + 0.5 * (spacing - target_spacing) * voxel_arr.shape[1:]
    return new_voxel_arr, target_spacing * np.ones_like(spacing) / downsampling, origin


def adjust_spacing_of_voxel_array_for_patch(voxel_arr: np.ndarray, spacing: np.ndarray, target_spacing: np.ndarray, adjusted_voxel_shapes: np.ndarray):
    """Adjuste spacing of voxel array for patch
    Args:
        voxel_arr (np.ndarray): voxel array. Shape is (W,H,D).
        spacing (np.ndarray): spacing. Shape is (3,).
        target_spacing (np.ndarray): target spacing. Shape is (3,).
        adjusted_voxel_shapes (np.ndarray): adjusted voxel shapes. Shape is (3,).

    Returns:
        new_voxel_arr (np.ndarray): new voxel array. Shape is (aW,aH,aD).
    """
    assert voxel_arr.ndim == 3
    assert spacing.ndim == 1
    assert target_spacing.ndim == 1

    grid_arr = tuple([spacing[j]*np.arange(voxel_arr.shape[j]) for j in range(3)]) #(3,W/H/D)
    new_grid_arr = np.stack(np.meshgrid(*[target_spacing[j]*np.arange(adjusted_voxel_shapes[j]) for j in range(3)], indexing="ij"), axis=-1) #(aW,aH,aD,3)
    new_voxel_arr = interpn(grid_arr, voxel_arr, new_grid_arr, method="linear", bounds_error=False, fill_value=0.0) #(aW,aH,aD)
    return new_voxel_arr



def transform_landmarks_to_downsampled_voxel(landmarks: np.ndarray, spacing: np.ndarray, origin: np.ndarray, downsampling: int = 1):
    """Transform landmarks to the downsampled voxel

    Args:
        landmarks (np.ndarray): (N,3:landmarks,3:xyz), the 3d coordinates of 3 landmarks
        spacing (np.ndarray): (N,3), the spacing of each voxel
        origin (np.ndarray): (N,3), the origin of each voxel
        downsampling (int): downsampling rate
    """
    landmarks = (landmarks - origin[:,None]) / spacing[:,None] / downsampling #(N,3,3)
    return landmarks


def load_split_dataframe(parent_dir: str, debug: bool = False):
    """Load split dataframe
    Args:
        parent_dir (str): parent directory
        debug (bool): If True, load only 5 samples.

    Returns:
        df (pd.DataFrame): dataframe
    """
    df = pd.read_csv(os.path.join(parent_dir, "cross_validation_split.csv"))
    if debug:
        first_indices = np.unique(df["fold"].values, return_index=True)[1]
        df = df.iloc[first_indices]
        df = df.reset_index(drop=True)
    total_path_list = df["DICOMPath"].values
    return df, total_path_list


def load_predicted_results(parent_dir: str, debug: bool = False, postname: str = "", path_list: Optional[np.ndarray] = None):
    """Load predicted results
    Args:
        parent_dir (str): parent directory
        debug (bool): If True, load only 5 samples.
        postname (str): postname

    Returns:
        predicted_landmarks (np.ndarray): predicted landmarks. Shape is (N,3:landmarks,3:xyz).
        path_list (np.ndarray): path list. Shape is (N,).
    """
    df = pd.read_csv(os.path.join(parent_dir, f"predicted_landmark{postname}.csv"))
    if path_list is not None:
        df = df.set_index("DICOMPath")
        df = df.loc[path_list]
        df.reset_index(inplace=True)
    elif debug:
        df = df.dropna(subset=["CTquality"])
        df = df[:5]
    path_list = df["DICOMPath"].values
    predicted_landmarks = df[["pRCC_x", "pRCC_y", "pRCC_z", "pLCC_x", "pLCC_y", "pLCC_z", "pNCC_x", "pNCC_y", "pNCC_z"]].values.reshape(-1,3,3) #(N,3,3)
    return predicted_landmarks, path_list


def get_patched_voxel_array_from_voxels(voxel_arr: np.ndarray, spacing: np.ndarray, origin: np.ndarray, predicted_landmarks: np.ndarray, target_spacing: Optional[float] = None, patch_size: int = 64):
    W_arr = voxel_arr.shape[1:] # (W,H,D)
    if target_spacing is None:
        target_spacing = spacing
    else:
        target_spacing = target_spacing * np.sign(spacing) # signed target spacing (N,3)
    adjusted_voxel_shapes = np.ceil(W_arr * spacing / target_spacing).astype(int) #(N,3)
    target_spacing = spacing * W_arr / adjusted_voxel_shapes #(N,3)
    difference_of_full_size = target_spacing * adjusted_voxel_shapes - spacing * W_arr #(N,3)
    predicted_landmarks = transform_landmarks_to_downsampled_voxel(predicted_landmarks, target_spacing, origin) #(N,3,3) at adjusted grid sp.
    patch_origin = compute_patch_origin(predicted_landmarks, patch_size, adjusted_voxel_shapes, difference_of_full_size) #(N,3,3)

    total_obs = []
    for arr, po_sample, s, ts, avs in zip(voxel_arr, patch_origin, spacing, target_spacing, adjusted_voxel_shapes):
        if np.any(s != ts):
            arr = adjust_spacing_of_voxel_array_for_patch(arr, s, ts, avs) #(aW,aH,aD)
        obs = []
        for po in po_sample:
            obs.append(arr[po[0]:po[0]+patch_size, po[1]:po[1]+patch_size, po[2]:po[2]+patch_size]) #(pW,pH,pD)
        total_obs.append(np.stack(obs)) #(3,pW,pH,pD)
    total_obs = np.stack(total_obs) #(N,3,pW,pH,pD)

    return total_obs, target_spacing, origin, patch_origin, adjusted_voxel_shapes, target_spacing


def get_patched_voxel_array(csv_path: str, path_list: Union[List[str], np.ndarray], predicted_landmarks: np.ndarray, patch_size: int, target_spacing: Optional[float] = None, using_prediction_for_computing_patch_origin: bool = True, voxel_shapes: List[int] = [512,512,320]):
    """Get patched voxel array
    Args:
        csv_path (str): path to csv file
        path_list (List[str]): list of path to voxel array
        predicted_landmarks (np.ndarray): (N,3,3), the predicted 3d coordinates of 3 landmarks at original sp.
        patch_size (int): patch size
        target_spacing (Optional[float]): target spacing
        using_prediction_for_computing_patch_origin (bool): whether to use prediction for computing patch origin
        voxel_shapes (List[int]): voxel shapes

    Returns:
        np.ndarray: (N,3,pW,pH,pD), the patched voxel array
    """
    if len(path_list) > 0:
        original_target_spacing = target_spacing
        landmarks, spacing, target_spacing, adjusted_voxel_shapes, origin = get_features_of_specific_paths(csv_path, path_list, target_spacing, voxel_shapes) #(N,3,3) at adjusted signed grid sp, (N,3) signed, (N,3) signed, (N,3), (N,3)
        if using_prediction_for_computing_patch_origin:
            predicted_landmarks = transform_landmarks_to_downsampled_voxel(predicted_landmarks, target_spacing, origin) #(N,3,3) at adjusted signed grid sp.
            patch_origin = compute_patch_origin(predicted_landmarks, patch_size, adjusted_voxel_shapes) # unsigned (N,3,3)
        else:
            patch_origin = compute_patch_origin(landmarks, patch_size, adjusted_voxel_shapes) # unsigned (N,3,3)
        landmarks -= patch_origin #(N,3,3) at patched adjusted grid sp.
        path_list = [os.path.join(csv_path.rsplit("/", 1)[0], path, "voxel_array.npy") for path in path_list]

        total_obs = []
        for po_sample, s, ts, avs, p in zip(patch_origin, spacing, target_spacing, adjusted_voxel_shapes, path_list):
            if os.path.exists(p.replace(".npy", f"_sp{original_target_spacing}.npy")):
                arr = np.load(p.replace(".npy", f"_sp{original_target_spacing}.npy")).astype("float32") #(W,H,D)
            else:
                arr = np.load(p).astype("float32").swapaxes(0,1) #(W,H,D)
                if np.any(s != ts):
                    arr = adjust_spacing_of_voxel_array_for_patch(arr, s, ts, avs) #(aW,aH,aD)
            obs = []
            for po in po_sample:
                obs.append(arr[po[0]:po[0]+patch_size, po[1]:po[1]+patch_size, po[2]:po[2]+patch_size]) #(pW,pH,pD)
            total_obs.append(np.stack(obs)) #(3,pW,pH,pD)
        total_obs = np.stack(total_obs) #(N,3,pW,pH,pD)
    else:
        total_obs = np.zeros((0,3,patch_size,patch_size,patch_size), dtype="float32")
        landmarks = np.zeros((0,3,3), dtype="float32")
        target_spacing = np.zeros((0,3), dtype="float32")
        origin = np.zeros((0,3), dtype="float32")
        patch_origin = np.zeros((0,3,3), dtype="int")
    return total_obs, landmarks, target_spacing, origin, patch_origin


def get_features_of_specific_paths(csv_path: str, path_list: Union[List[str], np.ndarray], target_spacing: Optional[float] = None, voxel_shapes: List[int] = [512,512,320]):
    """Get features of specific paths
    Args:
        csv_path (str): path to csv file
        path_list (List[str]): list of path to voxel array
        target_spacing (Optional[float]): target spacing
        voxel_shapes (List[int]): voxel shapes

    Returns:
        np.ndarray: landmarks (N,3,3)
        np.ndarray: spacing (N,3)
        np.ndarray: target spacing (N,3)
        np.ndarray: adjusted voxel shapes (N,3)
        np.ndarray: origin (N,3)
    """
    df = pd.read_csv(csv_path)
    if "Batch_dir" in df.columns:
        df["DICOMPath"] = df["Batch_dir"] + "/" + df["DICOMPath"]
    df = df.set_index("DICOMPath")
    df = df.loc[path_list]
    df = df.dropna(subset=["RCC_x", "RCC_y", "RCC_z", "LCC_x", "LCC_y", "LCC_z", "NCC_x", "NCC_y", "NCC_z"])
    landmarks = df[["RCC_x", "RCC_y", "RCC_z", "LCC_x", "LCC_y", "LCC_z", "NCC_x", "NCC_y", "NCC_z"]].values.astype("float32").reshape(-1, 3, 3) #(N,3,3)
    spacing = df[["spacing_x", "spacing_y", "spacing_z"]].values.astype("float32") #(N,3)
    if target_spacing is None:
        target_spacing = spacing
    else:
        target_spacing = target_spacing * np.sign(spacing) # signed target spacing (N,3)
    adjusted_voxel_shapes = np.ceil(np.array(voxel_shapes) * spacing / target_spacing).astype(int) #(N,3)
    target_spacing = spacing * np.array(voxel_shapes) / adjusted_voxel_shapes #(N,3)
    origin = df[["origin_x", "origin_y", "origin_z"]].values.astype("float32") #(N,3)
    landmarks = transform_landmarks_to_downsampled_voxel(landmarks, target_spacing, origin) #(N,3,3) at adjusted signed grid sp.
    return landmarks, spacing, target_spacing, adjusted_voxel_shapes, origin


def compute_patch_origin(predicted_landmarks: np.ndarray, patch_size: int, voxel_shapes: np.ndarray):
    """Compute patch origin
    Args:
        predicted_landmarks (np.ndarray): (N,3,3), the predicted 3d coordinates of 3 landmarks at adjusted grid sp.
        patch_size (int): patch size
        voxel_shapes (np.ndarray): (N,3), the voxel shapes at adjusted grid sp.

    Returns:
        np.ndarray: (N,3,3), the patch origin
    """
    patch_origin = (predicted_landmarks + 0.5).astype(int) - patch_size // 2 #(N,3,3)
    patch_terminal = patch_origin + patch_size #(N,3,3)
    patch_correction_vector = np.clip(-patch_origin, 0, None) + np.clip(voxel_shapes[:,None] - 1 - patch_terminal, None, 0) #(N,3,3)
    patch_origin += patch_correction_vector #(N,3,3)
    return patch_origin


def reconstruct_voxel_array_and_landmarks_for_unified_coordinate_from_voxel_array(voxel_arr: np.ndarray, spacing: np.ndarray, adjusted_voxel_shapes: np.ndarray, predicted_landmarks: np.ndarray, patch_origin: np.ndarray, patch_size: int, target_spacing: Optional[float] = None):
    """Reconstruct voxel array and landmarks for unified coordinate
    Args:
        path_list (Union[List[str], np.ndarray]): list of path to voxel array
        predicted_landmarks (np.ndarray): (N,3,3), the predicted 3d coordinates of 3 landmarks at adjusted grid sp.
        patch_origin (np.ndarray): (N,3,3), the patch origin
        patch_size (int): patch size
        target_spacing (Optional[float], optional): target spacing. Defaults to None.

    Returns:
        Tuple[List[np.ndarray], np.ndarray]: reconstructed voxel arrays (N,1,pW,pH,@D), reconstructed landmarks (N,3,3)
    """
    predicted_landmarks = predicted_landmarks + patch_origin #(N,3,3) at adjusted grid sp.
    reconstructed_patch_origin = patch_origin.min(axis=1) #(N,3)
    reconstructed_patch_terminal = patch_origin.max(axis=1) + patch_size #(N,3)
    W_arr = (reconstructed_patch_terminal - reconstructed_patch_origin).max(-1) #(N,)
    reconstructed_patch_origin = np.zeros_like(patch_origin[:,0]) #(N,3)
    reconstructed_patch_terminal = adjusted_voxel_shapes #(N,3)

    reconstructed_voxels = []
    for arr, rpo, rpt, s, ts, avs in zip(voxel_arr, reconstructed_patch_origin, reconstructed_patch_terminal, spacing, target_spacing, adjusted_voxel_shapes):
        if np.any(ts != s):
            arr = adjust_spacing_of_voxel_array_for_patch(arr, s, ts, avs) #(aW,aH,aD)
        arr = arr[rpo[0]:rpt[0], rpo[1]:rpt[1], rpo[2]:rpt[2]][None] #(1,pW,pH,pD)
        reconstructed_voxels.append(arr)
    return reconstructed_voxels, predicted_landmarks, W_arr


def reconstruct_voxel_array_and_landmarks_for_unified_coordinate(csv_path: str, path_list: Union[List[str], np.ndarray], predicted_landmarks: np.ndarray, patch_origin: np.ndarray, patch_size: int, target_spacing: Optional[float] = None, extracting_minimum_voxels: bool = False, voxel_shapes: List[int] = [512,512,320]) -> Tuple[List[np.ndarray], np.ndarray]:
    """Reconstruct voxel array and landmarks for unified coordinate
    Args:
        csv_path (str): path to csv file
        path_list (Union[List[str], np.ndarray]): list of path to voxel array
        predicted_landmarks (np.ndarray): (N,3,3), the predicted 3d coordinates of 3 landmarks at adjusted grid sp.
        patch_origin (np.ndarray): (N,3,3), the patch origin
        patch_size (int): patch size
        target_spacing (Optional[float], optional): target spacing. Defaults to None.
        extracting_minimum_voxels (bool, optional): extracting minimum voxels. Defaults to False.
        voxel_shapes (List[int], optional): voxel shapes. Defaults to [512,512,320].

    Returns:
        Tuple[List[np.ndarray], np.ndarray]: reconstructed voxel arrays (N,1,pW,pH,@D), reconstructed landmarks (N,3,3)
    """
    original_target_spacing = target_spacing
    landmarks, spacing, target_spacing, adjusted_voxel_shapes, _ = get_features_of_specific_paths(csv_path, path_list, target_spacing, voxel_shapes) #(N,3), (N,3), (N,3)
    predicted_landmarks = predicted_landmarks + patch_origin #(N,3,3) at adjusted grid sp.
    reconstructed_patch_origin = patch_origin.min(axis=1) #(N,3)
    reconstructed_patch_terminal = patch_origin.max(axis=1) + patch_size #(N,3)
    if extracting_minimum_voxels:
        reconstructed_landmarks = landmarks - reconstructed_patch_origin[:,None] #(N,3,3)
        reconstructed_predicted_landmarks = predicted_landmarks - reconstructed_patch_origin[:,None] #(N,3,3)
    else:
        W_arr = (reconstructed_patch_terminal - reconstructed_patch_origin).max(-1) #(N,)
        reconstructed_patch_origin = np.zeros_like(patch_origin[:,0]) #(N,3)
        reconstructed_patch_terminal = adjusted_voxel_shapes #(N,3)
        reconstructed_landmarks = landmarks #(N,3,3)
        reconstructed_predicted_landmarks = predicted_landmarks #(N,3,3)

    reconstructed_voxels = []
    path_list = [os.path.join(csv_path.rsplit("/", 1)[0], path, "voxel_array.npy") for path in path_list]
    for rpo, rpt, s, ts, avs, p in zip(reconstructed_patch_origin, reconstructed_patch_terminal, spacing, target_spacing, adjusted_voxel_shapes, path_list):
        if os.path.exists(p.replace(".npy", f"_sp{original_target_spacing}.npy")):
            arr = np.load(p.replace(".npy", f"_sp{original_target_spacing}.npy")).astype("float32")
        else:
            arr = np.load(p).astype("float32").swapaxes(0,1) #(W,H,D)
            if np.any(ts != s):
                arr = adjust_spacing_of_voxel_array_for_patch(arr, s, ts, avs) #(aW,aH,aD)
        arr = arr[rpo[0]:rpt[0], rpo[1]:rpt[1], rpo[2]:rpt[2]][None] #(1,pW,pH,pD)
        reconstructed_voxels.append(arr)

    return reconstructed_voxels, reconstructed_landmarks, reconstructed_predicted_landmarks, W_arr



def train_valid_test_split(data_list: List[Optional[np.ndarray]], val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 0):
    """Train valid test split
    Args:
        data_list (list): list of data
        val_ratio (float): valid ratio. Defaults to 0.1.
        test_ratio (float): test ratio. Defaults to 0.1.
        seed (int): seed. Defaults to 0.

    Returns:
        list: list of data
    """
    np.random.seed(seed)
    n_samples = len(data_list[0])

    val_num = int(val_ratio > 0) * max(math.ceil(n_samples * val_ratio), 2)
    test_num = math.ceil(n_samples * test_ratio)
    train_num = n_samples - val_num - test_num

    perm = np.random.permutation(n_samples)
    train_idx = perm[:train_num]
    val_idx = perm[train_num:train_num+val_num]
    test_idx = perm[train_num+val_num:]

    new_data_list = [[d[train_idx], d[val_idx], d[test_idx]] if d is not None else itertools.repeat(None) for d in data_list]
    new_data_list.append([train_idx, val_idx, test_idx])

    return new_data_list


def cross_validation_split(path_list: list, result_dir: str, n_splits: int = 5, test_ratio: float = 0.1, seed: int = 0) -> List[np.ndarray]:
    """Cross validation split
    Args:
        path_list (list): list of path to voxel array
        result_dir (str): result directory
        n_splits (int): number of splits. Defaults to 5.
        test_ratio (float): test ratio. Defaults to 0.1.
        seed (int): seed. Defaults to 0.

    Returns:
        list: list of np.ndarray
    """
    np.random.seed(seed)
    n_samples = len(path_list)
    test_num = int(test_ratio > 0) * max(math.ceil(n_samples * test_ratio), 2)

    perm = np.random.permutation(n_samples)
    cv_idx = perm[:-test_num]
    test_idx = perm[-test_num:]
    splits = np.array_split(cv_idx, n_splits)
    splits = [np.sort(s) for s in splits]

    # save split result
    df = pd.DataFrame(path_list, columns=["DICOMPath"])
    df["fold"] = -1
    for i, split in enumerate(splits):
        df.loc[split, "fold"] = i
    df.to_csv(os.path.join(result_dir, "cross_validation_split.csv"), index=False)

    return splits, test_idx


def correct_perturbed_voxel_and_landmarks(voxel_arr: np.ndarray, landmarks: np.ndarray, heatmaps: Optional[np.ndarray] = None, patch_size: int = 64, patch_perturbation: int = 0, local_patch_size: int = 1):
    """Correct perturbed voxel and landmarks
    Args:
        voxel_arr (np.ndarray): (N,3,W,H,D)
        landmarks (np.ndarray): (N,3,3)
        heatmaps (Optional[np.ndarray], optional): (N,3,w,h,d). Defaults to None.
        patch_size (int, optional): patch size. Defaults to 64.
        patch_perturbation (int, optional): patch perturbation. Defaults to 0.
        local_patch_size (int, optional): local patch size. Defaults to 1.
    """
    if patch_perturbation > 0:
        # adjust patches without over two-sides
        adjusted_voxel_arr = voxel_arr[:,:,patch_perturbation:patch_size+patch_perturbation,patch_perturbation:patch_size+patch_perturbation,patch_perturbation:patch_size+patch_perturbation] #(N,3,W,H,D)
        if heatmaps is not None:
            adjusted_heatmaps = heatmaps[:,:,patch_perturbation//local_patch_size:(patch_size+patch_perturbation)//local_patch_size,patch_perturbation//local_patch_size:(patch_size+patch_perturbation)//local_patch_size,patch_perturbation//local_patch_size:(patch_size+patch_perturbation)//local_patch_size] #(N,3,w,h,d)

        # compute adjusted samples
        landmarks *= local_patch_size
        adjusted_samples = np.array(np.where(np.any(np.any([landmarks < patch_perturbation, landmarks > patch_size + patch_perturbation], axis=0), axis=-1))).T #(n_over_samples,2)
        print("range of train landmarks: ", landmarks.min(), landmarks.max())

        # adjusted landmarks
        adjusted_delta = patch_perturbation + np.where(landmarks - patch_perturbation < 0, landmarks - patch_perturbation, 0) + np.where(landmarks - patch_perturbation > patch_size, landmarks - patch_perturbation - patch_size, 0) #(N,3,3)
        adjusted_delta = adjusted_delta.astype(int)
        adjusted_landmarks = landmarks - adjusted_delta # (N,3,3) at patch adjusted grid sp.
        adjusted_landmarks /= local_patch_size
        print("no adjusted: ", np.all(adjusted_delta == patch_perturbation))

        # adjust patches over two-sides
        for (i,j) in adjusted_samples:
            adjusted_voxel_arr[i,j] = voxel_arr[i,j,adjusted_delta[i,j,0]:adjusted_delta[i,j,0]+patch_size,adjusted_delta[i,j,1]:adjusted_delta[i,j,1]+patch_size,adjusted_delta[i,j,2]:adjusted_delta[i,j,2]+patch_size]
            if heatmaps is not None:
                adjusted_heatmaps[i,j] = heatmaps[i,j,adjusted_delta[i,j,0]//local_patch_size:(adjusted_delta[i,j,0]+patch_size)//local_patch_size,adjusted_delta[i,j,1]//local_patch_size:(adjusted_delta[i,j,1]+patch_size)//local_patch_size,adjusted_delta[i,j,2]//local_patch_size:(adjusted_delta[i,j,2]+patch_size)//local_patch_size]
    else:
        adjusted_voxel_arr = voxel_arr
        adjusted_heatmaps = heatmaps
        adjusted_landmarks = landmarks

    if heatmaps is None:
        adjusted_heatmaps = None

    return adjusted_voxel_arr, adjusted_landmarks, adjusted_heatmaps


def load_data(csv_path: str = "data/landmark.csv", downsampling: int = 4, handling_samples_over_max_length: str = "del", max_z_length: int = 320, target_spacing: Optional[Union[float, List[float], np.ndarray]] = None, heatmap_sigma: float = 1.0, balancing_weight_on: bool = False, debug: bool = False, return_heatmaps: bool = True):
    """Load voxel array and landmarks from csv file.
    Args:
        csv_path (str): Path to csv file.
        downsampling (int): Downsampling rate.
        handling_samples_over_max_length (str): How to handle samples whose z length is over max_z_length. "del" or "cut".
        max_z_length (int): Max z length.
        target_spacing (float or list of float): Target spacing.
        heatmap_sigma (float): Sigma of heatmap.
        gaussian_height (float): Height of gaussian.
        balancing_weight_on (bool): If True, compute weights for balancing.
        debug (bool): If True, load only 2 samples.
        return_heatmaps (bool): If True, return heatmaps.
    """
    voxel_arr, landmarks, spacing, origin, path_list = get_voxel_array(csv_path, downsampling, handling_samples_over_max_length, max_z_length, debug) # at original sp. (N,W,H,D), (N,3,3), (N,3), (N,3)
    if target_spacing is not None:
        voxel_arr, spacing, origin = adjust_spacing_of_voxel_array(voxel_arr, spacing, origin, target_spacing, downsampling) # at adjusted sp. (N,W,H,D), unsigned (N,3), adjusted unsigned (N,3)
    landmarks = transform_landmarks_to_downsampled_voxel(landmarks, spacing, origin, downsampling) # at adjusted downsampled sp. (N,3:landmarks,3:xyz)

    if return_heatmaps:
        heatmaps = get_heatmaps(landmarks, spacing, downsampling, voxel_arr.shape, heatmap_sigma) #(N,4,W,H,D)
        weights = compute_weights(heatmaps, balancing_weight_on) #(4,)/float
        return voxel_arr[:,None], heatmaps, spacing, origin, weights, landmarks, path_list
    else:
        return voxel_arr[:,None], spacing, origin, landmarks, path_list


def compute_adjusted_landmark_differences(csv_path: str, path_list: Union[List[str], np.ndarray], landmark_prediction: np.ndarray, voxel_shapes: np.ndarray, target_spacing: Union[float, List, np.ndarray] = 0.5, debug: bool = False):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["RCC_x", "RCC_y", "RCC_z", "LCC_x", "LCC_y", "LCC_z", "NCC_x", "NCC_y", "NCC_z"])
    df = df.set_index("DICOMPath")
    df = df.loc[path_list]
    landmarks = df[["RCC_x", "RCC_y", "RCC_z", "LCC_x", "LCC_y", "LCC_z", "NCC_x", "NCC_y", "NCC_z"]].values.astype("float32").reshape(len(df), 3, 3) #(N,3,3)
    spacing = df[["spacing_x", "spacing_y", "spacing_z"]].values.astype("float32") #(N,3)
    origin = df[["origin_x", "origin_y", "origin_z"]].values.astype("float32") #(N,3)

    if target_spacing is not None:
        origin = origin + 0.5 * (spacing - target_spacing) * voxel_shapes #(N,3)
        spacing = target_spacing * np.ones_like(spacing) #(N,3)
    landmarks = (landmarks - origin[:,None]) / spacing[:,None] #(N,3,3) at biadjusted sp.
    landmark_prediction = (landmark_prediction - origin[:,None]) / spacing[:,None] #(N,3,3) at biadjusted sp.
    landmark_differences = landmarks - landmark_prediction #(N,3,3) at biadjusted sp.

    return landmark_prediction, landmark_differences
  

def get_heatmaps(landmarks: np.ndarray, spacing: np.ndarray, downsampling: int, shapes: np.ndarray, sigma: float = 8.0):
    """Get heatmaps
    Calculate the Gaussian kernel in the 3d coordinates

    Args:
        landmarks (np.ndarray): (N,3,3), landmarks at adjusted downsampled sp.
        spacing (np.ndarray): (N,3), the spacing of each voxel at adjusted sp.
        origin (np.ndarray): (N,3), the origin of each voxel at adjusted sp.
        downsampling (int): downsampling rate
        shapes (np.ndarray): (4,), the downsampled shape of each voxel
        sigma (float, optional): sigma of gaussian kernel. Defaults to 1.0.
    """
    grid_arr = downsampling * (np.stack(np.meshgrid(np.arange(shapes[1]), np.arange(shapes[2]), np.arange(shapes[3]), indexing="ij"), axis=0)[None] + 0.5) * spacing[:,:,None,None,None] # at original sp. (N,3,W,H,D)

    fg_heatmaps = np.exp(-np.linalg.norm(grid_arr[:,None] - downsampling * (spacing[:,None] * landmarks)[:,:,:,None,None,None], axis=2)**2 / (2*sigma**2)) #(N,3,W,H,D)
    return fg_heatmaps
        

def get_patched_voxel_array_and_heatmaps(csv_path: str, path_list: Union[List[str], np.ndarray], predicted_landmarks: np.ndarray, patch_size: int, target_spacing: Optional[float] = None, downsampling: int = 1, heatmap_sigma: float = 8.0, using_prediction_for_computing_patch_origin: bool = True):
    """Get patched voxel array and heatmaps
    Args:
        csv_path (str): the path to the saved voxel array
        path_list (Union[List[str], np.ndarray]): the list of paths to the dicom files
        predicted_landmarks (np.ndarray): (N,3,3), the predicted landmarks at adjusted sp.
        patch_size (int): the size of the patch
        target_spacing (Optional[float], optional): the target spacing. Defaults to None.
        downsampling (int, optional): the downsampling rate. Defaults to 1.
        heatmap_sigma (float, optional): sigma of gaussian kernel. Defaults to 8.0.
        using_prediction_for_computing_patch_origin (bool, optional): If True, use the prediction for computing the patch origin. Defaults to True.
    """
    voxel_arr, landmarks, spacing, origin, patch_origin = get_patched_voxel_array(csv_path, path_list, predicted_landmarks, patch_size, target_spacing, using_prediction_for_computing_patch_origin) #(N,3,W,H,D), (N,3,3), (N,3), (N,3), (N,3,3)
    if len(voxel_arr) > 0:
        heatmaps = get_heatmaps(landmarks, spacing, downsampling, voxel_arr[:,0].shape, heatmap_sigma) #(N,3,W,H,D)
    else:
        heatmaps = np.zeros((0,3,patch_size,patch_size,patch_size), dtype="float32")
    return voxel_arr, landmarks, spacing, origin, patch_origin, heatmaps


def compute_weights(heatmaps: np.ndarray, balancing_weight_on: bool = True):
    """Compute weights
    Args:
        heatmaps (np.ndarray): (N,4,W,H,D), the heatmaps
        balancing_weight_on (bool, optional): If True, compute the balancing weight. Defaults to True.
    """
    if balancing_weight_on:
        class_weights = 1 - heatmaps.mean()
        return class_weights
    else:
        return None


def compute_perturbed_voxel_arr_and_heatmaps(voxel_arr: torch.Tensor, heatmaps: torch.Tensor, patch_size: int = 64):
    """Compute perturbed voxel array and landmarks
    Args:
        voxel_arr (torch.Tensor): (N,3,W+pp,H+pp,D+pp), the voxel array
        heatmaps (torch.Tensor): (N,3,W+pp,H+pp,D+pp), the heatmaps
        patch_size (int, optional): the size of the patch. Defaults to 64.
    """
    n_samples, n_channels, W, _, _ = voxel_arr.shape
    voxel_start_position = torch.randint(0, W - patch_size, (n_samples, n_channels, 3)) #(N,3,3)
    perturbed_voxel_arr = torch.zeros((n_samples, n_channels, patch_size, patch_size, patch_size), dtype=voxel_arr.dtype, device=voxel_arr.device)
    perturbed_heatmaps = torch.zeros((n_samples, n_channels, patch_size, patch_size, patch_size), dtype=heatmaps.dtype, device=heatmaps.device)

    for i in range(n_samples):
        for j in range(n_channels):
            perturbed_voxel_arr[i,j] = voxel_arr[i,j,voxel_start_position[i,j,0]:voxel_start_position[i,j,0]+patch_size, voxel_start_position[i,j,1]:voxel_start_position[i,j,1]+patch_size, voxel_start_position[i,j,2]:voxel_start_position[i,j,2]+patch_size]
            perturbed_heatmaps[i,j] = heatmaps[i,j,voxel_start_position[i,j,0]:voxel_start_position[i,j,0]+patch_size, voxel_start_position[i,j,1]:voxel_start_position[i,j,1]+patch_size, voxel_start_position[i,j,2]:voxel_start_position[i,j,2]+patch_size]

    return perturbed_voxel_arr, perturbed_heatmaps