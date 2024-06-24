from typing import List, Union, Optional
import os, math, itertools, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interpn
from sklearn.cluster import AgglomerativeClustering
import torch
import torch.nn.functional as F
import logtools, data, utils


def generate_results(args: argparse.Namespace, device: torch.device, m: torch.nn.Module, logger: logtools.BaseLogger, global_phase: bool, total_path_list: List[str], voxel_arr_list: List[np.ndarray], heatmaps_list: List[np.ndarray],spacing_list: List[np.ndarray], origin_list: List[np.ndarray], patch_origin_list: List[np.ndarray], landmarks_list: List[np.ndarray], path_list_list: List[List[str]], predicted_landmarks: Optional[np.ndarray], idx_list: List[np.ndarray], phase_list = ["train", "valid", "test"], plot_on: bool = True, post_name: str = ""):
    m.eval()

    if args.patch_perturbation > 0 and (not global_phase) and phase_list[0] == "train":
        voxel_arr_list[0], landmarks_list[0], spacing_list[0], origin_list[0], patch_origin_list[0], heatmaps_list[0] = data.get_patched_voxel_array_and_heatmaps(args.data_path, total_path_list[idx_list[0]], predicted_landmarks[idx_list[0]], args.patch_size, args.spacing, args.downsampling, args.heatmap_sigma, args.gaussian_height, args.heatmap_type, args.norm_order, args.power_cost_on, True) #(N,3,W,H,D), (N,3,3), (N,3), (N,3), (N,3,3), (N,3,W,H,D)

    total_predicted_landmarks = np.zeros([len(total_path_list), 3, 3]) #(N,3,3)
    W_arr = itertools.repeat(None)

    for voxel_arr, heatmaps, spacing, origin, patch_origin, landmarks, path_list, indices, phase in zip(voxel_arr_list, heatmaps_list, spacing_list, origin_list, patch_origin_list, landmarks_list, path_list_list, idx_list, phase_list):
        if len(voxel_arr) > 0:
            predicted_heatmaps = predict_heatmaps(m, device, voxel_arr, args.batch_size, args.loss_function, args.heatmap_type) #(N,4,W,H,D)
            predicted_landmarks = predict_landmarks(predicted_heatmaps) #(N,3,3) at adjusted downsampled sp.
            distance_error = compute_error_of_landmark_displacement(logger, landmarks, predicted_landmarks, args.data_path, path_list, spacing, patch_origin, args.downsampling, args.result_dir, phase, post_name) #(N,)
            total_predicted_landmarks[indices] = compute_predicted_landmark_at_original_space(predicted_landmarks, spacing, origin, patch_origin, args.downsampling) #(N,3,3)
            
            if plot_on:
                boxplot_by_quality(args.result_dir, logger, phase)
                if phase=="train":
                    worst_indices = np.argsort(-distance_error)[:args.number_of_training_plots]
                    voxel_arr, heatmaps, predicted_heatmaps, landmarks, predicted_landmarks, path_list, spacing = [v[worst_indices] if v is not None else None for v in [voxel_arr, heatmaps, predicted_heatmaps, landmarks, predicted_landmarks, path_list, spacing]]
                    if patch_origin is not None:
                        patch_origin = patch_origin[worst_indices]
                plot_heatmaps(logger, args.result_dir, heatmaps, predicted_heatmaps, landmarks, path_list, phase)
                if patch_origin is not None:
                    voxel_arr, landmarks, predicted_landmarks, W_arr = data.reconstruct_voxel_array_and_landmarks_for_unified_coordinate(args.data_path, path_list, predicted_landmarks, patch_origin, args.patch_size, args.spacing) #(N,3,W,H,D), (N,3,3)
                plot_landmarks_at_slanted_pixels(voxel_arr, predicted_landmarks, path_list, args.downsampling * spacing, W_arr, landmarks, logger, args.result_dir, global_phase, phase)
    save_predicted_landmark(total_predicted_landmarks, args.data_path, total_path_list, args.result_dir, post_name)



def generate_results_for_landmark_regression(args: argparse.Namespace, device: torch.device, m: torch.nn.Module, logger: logtools.BaseLogger, total_path_list: List[str], voxel_arr_list: List[np.ndarray], predicted_landmarks_list: List[np.ndarray],spacing_list: List[np.ndarray], origin_list: List[np.ndarray], patch_origin_list: List[np.ndarray], landmarks_list: List[np.ndarray], path_list_list: List[List[str]], predicted_landmarks: Optional[np.ndarray], idx_list: List[np.ndarray], phase_list = ["train", "valid", "test"], plot_on: bool = True, post_name: str = ""):
    m.eval()

    if args.patch_perturbation > 0 and phase_list[0] == "train":
        voxel_arr_list[0], landmarks_list[0], spacing_list[0], origin_list[0], patch_origin_list[0], predicted_landmarks_list[0], _ = data.get_patched_voxel_array_and_landmarks(args.data_path, total_path_list[idx_list[0]], predicted_landmarks[idx_list[0]], args.patch_size, args.spacing, args.downsampling, True) #(N,3,W,H,D), (N,3,3), (N,3), (N,3), (N,3,3), (N,3,W,H,D)


    total_predicted_landmarks = np.zeros([len(total_path_list), 3, 3]) #(N,3,3)
    W_arr = itertools.repeat(None)

    for voxel_arr, spacing, origin, patch_origin, landmarks, predicted_landmarks, path_list, indices, phase in zip(voxel_arr_list, spacing_list, origin_list, patch_origin_list, landmarks_list, predicted_landmarks_list, path_list_list, idx_list, phase_list):
        if len(voxel_arr) > 0:
            predicted_landmark_differences = predict_local_landmark_differences(m, device, voxel_arr, args.batch_size) #(N,3,3)
            predicted_landmarks = predicted_landmarks + predicted_landmark_differences #(N,3,3)
            distance_error = compute_error_of_landmark_displacement(logger, landmarks, predicted_landmarks, args.data_path, path_list, spacing, patch_origin, args.downsampling, args.result_dir, None, phase, post_name) #(N,)
            total_predicted_landmarks[indices] = compute_predicted_landmark_at_original_space(predicted_landmarks, spacing, origin, patch_origin, args.downsampling) #(N,3,3)

            if plot_on:
                boxplot_by_quality(args.result_dir, logger, phase)
                if phase=="train":
                    worst_indices = np.argsort(-distance_error)[:args.number_of_training_plots]
                    voxel_arr, predicted_landmarks, landmarks, spacing, path_list = [v[worst_indices] for v in [voxel_arr, predicted_landmarks, landmarks, spacing, path_list]]
                    if patch_origin is not None:
                        patch_origin = patch_origin[worst_indices]
                if patch_origin is not None:
                    voxel_arr, landmarks, predicted_landmarks, W_arr = data.reconstruct_voxel_array_and_landmarks_for_unified_coordinate(args.data_path, path_list, predicted_landmarks, patch_origin, args.patch_size, args.spacing) #(N,3,W,H,D), (N,3,3)
                plot_landmarks_at_slanted_pixels(voxel_arr, predicted_landmarks, path_list, args.downsampling * spacing, W_arr, landmarks, logger, args.result_dir, False, phase)
    save_predicted_landmark(total_predicted_landmarks, args.data_path, total_path_list, args.result_dir, post_name)



def predict_heatmaps(m: torch.nn.Module, device: torch.device, voxel_arr: np.ndarray, batch_size: int = 8, loss_function: str = "MSELoss", heatmap_type: str = "joint"):
    """Predict heatmaps from voxel_arr.
    Args:
        m (torch.nn.Module): model
        device (torch.device): device
        voxel_arr (np.ndarray): voxel array (N,3,W,H,D)
        batch_size (int, optional): batch size. Defaults to 8.

    Returns:
        np.ndarray: heatmaps (N,4,W,H,D)
    """
    # set dataloader
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(voxel_arr).float())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # predict
    m.eval()
    heatmaps = []
    for x in loader:
        x = x[0].to(device)
        y_hat = m(x) #j/s:(bs,4/3,W,H,D),f:(bs,W*H*D,3)
        if heatmap_type == "s":
            if loss_function not in ["GWL", "IGWL"]:
                y_hat = F.sigmoid(y_hat)
        elif loss_function not in ["MSE", "L1", "SL1"]:
            y_hat = F.softmax(y_hat, dim=1)
            if heatmap_type in ["f", "d"]:
                W, H, D = x.shape[2:]
                y_hat = y_hat.transpose(1,2) #(bs,3,W*H*D)
                y_hat = y_hat.reshape(y_hat.shape[0], 3, W, H, D) #(bs,3,W,H,D)
        heatmaps.append(y_hat.detach().cpu().numpy())
    
    heatmaps = np.concatenate(heatmaps, axis=0) #j/s,f,d:(N,4/3,W,H,D)
    return heatmaps


def predict_landmarks(heatmaps: np.ndarray, spacing: Optional[np.ndarray] = None, origin: Optional[np.ndarray] = None, downsampling: int = 4):
    """Predict landmarks from heatmaps.
    Args:
        heatmaps (np.ndarray): heatmaps (N,4,W,H,D)

    Returns:
        np.ndarray: landmarks (N,3,3)
    """
    _, _, W, H, D = heatmaps.shape

    # calculate landmarks
    argmaxed_heatmap = np.argmax(heatmaps[:,:3].reshape(heatmaps.shape[0], 3, W*H*D), axis=-1) #(N,3)
    landmark_prediction = np.stack([argmaxed_heatmap // (H*D), (argmaxed_heatmap % (H*D)) // D, argmaxed_heatmap % D], axis=-1).astype(float) + 0.5 #(N,3,3)

    if spacing is not None and origin is not None:
        landmark_prediction = origin[:,None,:] + spacing[:,None,:] * downsampling * landmark_prediction #(N,3,3) at original sp.
    return landmark_prediction


def plot_heatmaps(logger: logtools.BaseLogger, result_dir: str, heatmaps: np.ndarray, predicted_heatmaps: np.ndarray, landmarks: np.ndarray, path_list: Optional[Union[List[str], np.ndarray]] = None, prename: str = "valid"):
    """Plot heatmaps.
    Args:
        logger (logtools.BaseLogger): logger
        result_dir (str): result directory
        heatmaps (np.ndarray): heatmaps (N,4,W,H,D)
        landmarks (np.ndarray): landmarks (N,3,3)
        path_list (Optional[Union[List[str], np.ndarray]], optional): path list. Defaults to None.
        prename (str, optional): prename. Defaults to "valid".
    """
    if heatmaps is None:
        return
    _, _, _, H, D = heatmaps.shape

    # plot heatmaps
    if path_list is not None:
        for i, p in enumerate(path_list):
            try:
                fig, ax = plt.subplots(2, 3*3, figsize=(3*3*5*(2*D+H)/(3*H), 2*5), width_ratios=3*[D,D,H])
                for j, h in enumerate([heatmaps, predicted_heatmaps]):
                    vmin, vmax = h.min(), h.max()
                    for k, landmark_name in enumerate(["RCC", "LCC", "NCC"]):
                        for l, axis_name in enumerate(["Coronal", "Saggital", "Axial"]):
                            ax[j, k*3+l].imshow(np.take(h[i, k], round(landmarks[i, k, l]), axis=l), vmin=vmin, vmax=vmax)
                            ax[j, k*3+l].set_axis_off()
                            if j == 0:
                                ax[j, k*3+l].set_title(f"{landmark_name} {axis_name}")
                            else:
                                ax[j, k*3+l].set_title(" ")
                    ax[j, 0].set_ylabel("Ground Truth" if j == 0 else "Predicted")
                fig.tight_layout()
                p = utils.exclude_japanese(p)
                fig.savefig(os.path.join(result_dir, "{}_{}_heatmap.pdf".format(prename, p)), bbox_inches="tight")
                logger.log_figure("{}_{}_heatmap".format(prename, p), fig)
            except:
                print("Failed to plot heatmap for {}".format(p))


def ensemble_landmark_predictions(landmark_prediction_list: List[np.ndarray], ensemble_method: str, distance_threshold: float = 5.0):
    """Ensemble landmark prediction.
    Args:
        landmark_prediction_list (List[np.ndarray]): landmark predictions list (*,N,3,3)
        ensemble_method (str): ensemble method
    Returns:
        np.ndarray: landmark prediction (N,3,3)
    """
    if len(landmark_prediction_list) == 1:
        landmark_prediction = landmark_prediction_list[0]
    else:
        if ensemble_method == "mean":
            landmark_prediction = np.mean(landmark_prediction_list, axis=0) #(N,3,3)
        elif ensemble_method == "median":
            landmark_prediction = np.median(landmark_prediction_list, axis=0) #(N,3,3)
        elif ensemble_method == "agglomerative":
            landmark_prediction_arr = np.stack(landmark_prediction_list, axis=0) #(B,N,3,3)
            landmark_prediction = []
            for i in range(landmark_prediction_list[0].shape[0]):
                lp = []
                for j in range(landmark_prediction_list[0].shape[1]):
                    cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, compute_full_tree=True).fit_predict(landmark_prediction_arr[:,i,j,:]) #(B,)
                    maximum_cluster = np.argmax(np.bincount(cluster))
                    lp.append(landmark_prediction_arr[cluster==maximum_cluster,i,j,:].mean(axis=0)) #(3,)
                lp = np.stack(lp, axis=0) #(3,3)
                landmark_prediction.append(lp)
            landmark_prediction = np.stack(landmark_prediction, axis=0) #(N,3,3)
        else:
            raise NotImplementedError
    return landmark_prediction


def predict_local_landmark_differences(m: torch.nn.Module, device: torch.device, patches: np.ndarray, batch_size: int = 4):
    """Predict landmark differences.
    Args:
        m (torch.nn.Module): model
        device (torch.device): device
        patches (np.ndarray): patches (N,3,pW,pH,pD)
        batch_size (int, optional): batch size. Defaults to 4.

    Returns:
        np.ndarray: landmark differences (N,3,3)
    """
    # set dataloader
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(patches).float())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # predict
    m.eval()
    landmark_differences = []
    for x in loader:
        x = x[0].to(device)
        y_hat = m(x) #(bs,3,3)
        landmark_differences.append(y_hat.detach().cpu().numpy())
    
    landmark_differences = np.concatenate(landmark_differences, axis=0) #(N,3,3)

    return landmark_differences


def plot_landmarks_and_predicted_landmarks_in_patch(logger: logtools.BaseLogger, result_dir: str, patches: np.ndarray, landmark_differences: np.ndarray, predicted_landmark_differences: np.ndarray, path_list: list, prename: str = "valid"):
    """Plot landmarks and predicted landmarks in patch.
    Args:
        logger (logtools.BaseLogger): logger
        result_dir (str): result directory
        patches (np.ndarray): patches (N,3,pW,pH,pD)
        landmark_differences (np.ndarray): landmark differences (N,3,3)
        predicted_landmark_differences (np.ndarray): predicted landmark differences (N,3,3)
        path_list (list): path list
        prename (str, optional): prename. Defaults to "valid".
    """
    # plot landmarks and predicted landmarks in patch
    patch_shapes = np.array(patches.shape[2:])
    axis_names = ["x", "y", "z"]

    for i, p in enumerate(path_list):
        fig, ax = plt.subplots(3,3,figsize=(9,9))
        for j, landmark_name in enumerate(["RCC", "LCC", "NCC"]):
            for k, axis_name in enumerate(["Coronal", "Sagittal", "Axial"]):
                ax[j,k].imshow(np.take(patches[i,j], int(landmark_differences[i,j,k] + patch_shapes[k]/2), axis=k), cmap="gray")
                idx2 = int(k==0)
                idx1 = int(k!=2) + 1
                ax[j,k].scatter(patch_shapes[idx1]/2, patch_shapes[idx2]/2, marker="x", color="blue", label="first guess")
                ax[j,k].scatter(patch_shapes[idx1]/2 + predicted_landmark_differences[i,j,idx1], patch_shapes[idx2]/2 + predicted_landmark_differences[i,j,idx2], marker="x", color="green", label="prediction")
                ax[j,k].scatter(patch_shapes[idx1]/2 + landmark_differences[i,j,idx1], patch_shapes[idx2]/2 + landmark_differences[i,j,idx2], marker="x", color="red", label="ground truth")
                ax[j,k].set_xlabel(f"{axis_names[idx1]} (px)")
                ax[j,k].set_ylabel(f"{axis_names[idx2]} (px)")
                ax[j,k].set_title(f"{landmark_name} ({axis_name})")
        fig.tight_layout()
        p = utils.exclude_japanese(p)
        fig.savefig(os.path.join(result_dir, f"{prename}_{p}_landmarks.pdf"), bbox_inches="tight")
        logger.log_figure(f"{prename}-{p}-landmarks", fig)


def compute_distance_metrics(logger: logtools.BaseLogger, landmark_differences: np.ndarray, predicted_landmark_differences: np.ndarray, spacing: np.ndarray, prename: str = "valid"):
    """Compute distance metrics.
    Args:
        logger (logtools.BaseLogger): logger
        landmark_differences (np.ndarray): landmark differences (N,3,3)
        predicted_landmark_differences (np.ndarray): predicted landmark differences (N,3,3)
        spacing (np.ndarray): spacing (N,3)
        prename (str, optional): prename. Defaults to "valid".
    """
    distance_error = np.linalg.norm(spacing[:,None] * (landmark_differences - predicted_landmark_differences), axis=2) #(N,3)
    logger.log_metrics({f"{prename}_{lname}_RMSE":e for e, lname in zip(np.sqrt(np.mean(distance_error**2, axis=0)), ["RCC", "LCC", "NCC"])})
    logger.log_metrics({f"{prename}_{lname}_WDE":e for e, lname in zip(np.max(distance_error, axis=0), ["RCC", "LCC", "NCC"])})


def get_predicted_results(m: torch.nn.Module, device: torch.device, voxel_arr: np.ndarray, spacing: np.ndarray, origin: np.ndarray, args: argparse.ArgumentParser, patch_origin: Optional[np.ndarray] = None):
    m.load_state_dict(torch.load(os.path.join(args.result_dir, "state_dict.pth"), map_location=device))
    m.to(device)
    m.eval()

    # predict
    predicted_heatmaps = predict_heatmaps(m, device, voxel_arr, args.batch_size, args.loss_function, args.heatmap_type) # (N,3,W,H,D)
    ad_predicted_landmarks = predict_landmarks(predicted_heatmaps) # (N,3,3) at adjusted downsampled sp.
    predicted_landmarks = compute_predicted_landmark_at_original_space(ad_predicted_landmarks, spacing, origin, patch_origin, args.downsampling) # (N,3,3) at original sp.

    return predicted_landmarks, ad_predicted_landmarks


def generate_final_prediction_results(voxel_arr: np.ndarray, spacing: np.ndarray, path_arr: np.ndarray, predicted_landmarks: np.ndarray, ad_predicted_landmarks: np.ndarray, outputs: list, local_phase: bool, g_voxel_arr: np.ndarray, downsampling: int, child_downsampling: Optional[int] = None, adjusted_voxel_shapes: Optional[np.ndarray] = None, target_spacing: Optional[np.ndarray] = None, patch_origin: Optional[np.ndarray] = None, patch_size: Optional[int] = None) -> dict:
    """Generate final prediction results.
    Args:
        voxel_arr (np.ndarray): voxel array (N,1,W,H,D)
        spacing (np.ndarray): spacing (N,3)
        path_arr (np.ndarray): path array (N,)
        predicted_landmarks (np.ndarray): predicted landmarks at original sp. (N,3,3)
        ad_predicted_landmarks (np.ndarray): predicted landmarks at adjusted downsampled sp. (N,3,3)
        outputs (list): outputs
        local_phase (bool): local phase
        g_voxel_arr (np.ndarray): voxel array at global phase (N,1,W,H,D)
        downsampling (int): downsampling
        child_downsampling (Optional[int], optional): downsampling at child phase. Defaults to None.
        adjusted_voxel_shapes (Optional[np.ndarray], optional): adjusted voxel shapes. Defaults to None.
        target_spacing (Optional[np.ndarray], optional): target spacing. Defaults to None.
        patch_origin (Optional[np.ndarray], optional): patch origin. Defaults to None.
        patch_size (Optional[int], optional): patch size. Defaults to None.
    
    Returns:
        dict: final prediction results
    """
    result_dict = {"predicted_landmarks": predicted_landmarks}
    if "n" in outputs:
        predicted_norm_vectors = compute_norm_vector(ad_predicted_landmarks) # (N,3)
        result_dict["predicted_norm_vectors"] = predicted_norm_vectors

    if "p" in outputs:
        if local_phase:
            r_voxel_arr, po_predicted_landmarks, W_arr = data.reconstruct_voxel_array_and_landmarks_for_unified_coordinate_from_voxel_array(voxel_arr, spacing, adjusted_voxel_shapes, ad_predicted_landmarks, patch_origin, patch_size, target_spacing)
            plot_landmarks_at_slanted_pixels(r_voxel_arr, po_predicted_landmarks, path_arr, child_downsampling * spacing, W_arr)
        else:
            plot_landmarks_at_slanted_pixels(g_voxel_arr, ad_predicted_landmarks, path_arr, downsampling * spacing)
    
    return result_dict


def plot_landmarks_at_slanted_pixels(voxel_arr: np.ndarray, predicted_landmark: np.ndarray, path_list: Union[np.ndarray, List[str]], spacing: np.ndarray, W_arr: Optional[np.ndarray] = None, landmarks: Optional[np.ndarray] = None, logger: Optional[logtools.BaseLogger] = None, result_dir: Optional[str] = None, zooming_on: bool = True, prename: str = "", text_fontsize: int = 15) -> None:
    """
    Args:
        voxel_arr (np.ndarray): voxel array (N,1,W,H,D)
        predicted_landmark (np.ndarray): predicted landmark at adjusted downsampled sp. (N,3,3)
        path_list (Union[np.ndarray, List[str]]): path list (N,)
        spacing (np.ndarray): spacing (N,3)
        W_arr (Optional[np.ndarray], optional): W array (N,). Defaults to None.
        landmarks (np.ndarray): landmarks at adjusted downsampled sp. (N,3,3)
        logger (logtools.BaseLogger): logger
        result_dir (str): result directory
        prename (str, optional): prename. Defaults to "".
    """
    if landmarks is None:
        landmarks = itertools.repeat(None)
    if W_arr is None:
        W_arr = itertools.repeat(None)

    for voxel, l, pl, p, s, w in zip(voxel_arr, landmarks, predicted_landmark, path_list, spacing, W_arr):
        # true landmarks exist
        if l is not None:
            fig, ax = plt.subplots(1+int(zooming_on),2,figsize=(6,3*(1+int(zooming_on))))

            for j, l1, l2, clist in zip(range(2), [l, pl], [pl, l], [["cyan", "tab:red"], ["tab:red", "cyan"]]):
                # compute slanted pixels and landmarks
                slanted_pixel_arr, (slanted_l1, slanted_l2), (_, projected_distance) = compute_slanted_pixels_and_landmarks(voxel[0], [l1, l2], w) #(W,H),(3,2),(3,2), (3,)

                if zooming_on:
                    # zooming region
                    min_xy, max_xy = compute_zooming_region([slanted_l1, slanted_l2], slanted_pixel_arr.shape[0])

                    # plot at slanted plane
                    for k, spa, sl1, sl2 in zip(range(2), [slanted_pixel_arr, slanted_pixel_arr[min_xy[0]:max_xy[0], min_xy[1]:max_xy[1]]], [slanted_l1, slanted_l1 - min_xy], [slanted_l2, slanted_l2 - min_xy]):
                        ax[k,j].imshow(spa.T, cmap="gray")
                        ax[k,j].scatter(*sl1.T, color=clist[0], s=20, marker="x")
                        ax[k,j].scatter(*sl2.T, color=clist[1], s=20, marker="x")
                        ax[k,j].set_axis_off()

                        # text of projected distance
                        for i in range(3):
                            ax[k,j].text(*sl2[i], f"{s[0]*projected_distance[i]:.1f}", color=clist[1], fontsize=text_fontsize)
                else:
                    ax[j].imshow(slanted_pixel_arr.T, cmap="gray")
                    ax[j].scatter(*slanted_l1.T, color=clist[0], s=20, marker="x")
                    ax[j].scatter(*slanted_l2.T, color=clist[1], s=20, marker="x")
                    ax[j].set_axis_off()

                    # text of projected distance
                    for i in range(3):
                        ax[j].text(*slanted_l2[i], f"{s[0]*projected_distance[i]:.1f}", color=clist[1], fontsize=text_fontsize)

            # log plot
            fig.tight_layout()
            p = utils.exclude_japanese(p)
            fig.savefig(os.path.join(result_dir, f"{prename}_{p}_landmark_prediction.pdf"), bbox_inches="tight")
            logger.log_figure(f"{prename} {p} landmark prediction", fig)
        else:
            fig, ax = plt.subplots(1,1,figsize=(3,3))

            # compute slanted pixels and landmarks
            slanted_pixel_arr, (slanted_l, ), _ = compute_slanted_pixels_and_landmarks(voxel[0], [pl], w) #(W,H),(3,2)

            # for slanted axis
            ax.imshow(slanted_pixel_arr.T, cmap="gray")
            ax.scatter(*slanted_l.T, color="tab:red", s=20, marker="x")
            ax.set_axis_off()

            # save
            fig.tight_layout()
            fig.savefig(os.path.join(p, f"landmark_prediction.pdf"), bbox_inches="tight")


def compute_slanted_pixels_and_landmarks(voxel_arr: np.ndarray, landmarks: List[np.ndarray], W: Optional[int] = None) -> np.ndarray:
    """
    Args:
        voxel_arr (np.ndarray): voxel array (W,H,D)
        landmarks (List[np.ndarray]): list of landmarks at adjusted downsampled sp. (2,3,3) or (1,3,3)

    Returns:
        np.ndarray: slanted voxel array (W,H)
        List[np.ndarray]: slanted landmarks [(3,2)]
        List[np.ndarray]: projected distance [(3,)]
    """
    plane_params = np.linalg.pinv(landmarks[0]).sum(axis=-1) #(3,)

    if W is None:
        W = int(math.sqrt(3) / math.sqrt(2) * max(voxel_arr.shape))

    norm_vector = plane_params / np.linalg.norm(plane_params) #(3,)
    tangent_vector_x = np.array([norm_vector[2], 0, -norm_vector[0]]) #(3,)
    tangent_vector_y = np.array([0, norm_vector[2], -norm_vector[1]]) #(3,)
    tangent_vector_x /= np.linalg.norm(tangent_vector_x) #(3,)
    tangent_vector_y /= np.linalg.norm(tangent_vector_y) #(3,)

    # compute the angle of two tangent vectors
    theta_xy = np.arccos(tangent_vector_x @ tangent_vector_y) #(1,)
    # acute_sign = np.sign(theta_xy - np.pi/2) #(1,)
    norm_sign = np.sign(norm_vector[2]) #(1,)
    norm_sign = 1 if norm_sign==0 else norm_sign
    adjusted_angle = norm_sign * (theta_xy/2 - np.pi/4) #(1,)

    # rotate tangent vectors
    Rx = compute_Rodrigues_rotation_matrix(norm_vector, adjusted_angle) #(3,3)
    Ry = compute_Rodrigues_rotation_matrix(norm_vector, -adjusted_angle) #(3,3)
    tangent_vector_x = Rx @ tangent_vector_x #(3,)
    tangent_vector_y = Ry @ tangent_vector_y #(3,)

    # compute barycenter of landmarks
    barycenter = landmarks[0].mean(axis=0) #(3,)

    # compute slanted pixel array
    original_grid_arr = tuple([np.arange(s).astype(float) for s in voxel_arr.shape]) # (W,H,D)
    grid_x, grid_y = np.meshgrid(np.arange(W)-W//2, np.arange(W)-W//2, indexing="ij") #(W,H), (W,H)
    slanted_grid_arr = grid_x[...,None]*tangent_vector_x + grid_y[...,None]*tangent_vector_y + barycenter #(W,H,3)
    slanted_pixel_arr = interpn(original_grid_arr, voxel_arr, slanted_grid_arr, method="linear", bounds_error=False, fill_value=0) #(W,H)

    # plane rotation matrix for landmark coordinates
    phi = np.sign(norm_vector[2]) * np.arccos(norm_vector[2]) #(1,)
    axis_of_rotation = np.array([norm_vector[1], -norm_vector[0], 0]) #(3,)
    axis_of_rotation /= np.linalg.norm(axis_of_rotation) #(3,)
    R = compute_Rodrigues_rotation_matrix(axis_of_rotation, phi) #(3,3)
    tangent_vector_x = R @ tangent_vector_x #(3,)
    tangent_vector_y = R @ tangent_vector_y #(3,)
    projected_tangent_matrix = np.stack([tangent_vector_x[:2], tangent_vector_y[:2]], axis=1) #(2,2)
    inv_projected_tangent_matrix = np.linalg.pinv(projected_tangent_matrix) #(2,2)

    # compute landmark in slanted pixels
    slanted_landmarks = []
    projected_distance = []
    for l in landmarks:
        foot_of_perpendicular_line = l + (1 - (plane_params*l).sum(axis=1)) / np.linalg.norm(plane_params)**2 * plane_params #(3:landmark,3:xyz)
        foot_of_perpendicular_line = R @ (foot_of_perpendicular_line - barycenter).T #(3:xyz,3:landmark)
        slanted_landmarks.append((inv_projected_tangent_matrix @ foot_of_perpendicular_line[:2]).T + np.array([W//2, W//2])) #(3:landmark,2:xy)
        projected_distance.append(np.absolute(1 - (plane_params*l).sum(axis=-1)) / np.linalg.norm(plane_params, axis=-1)) #(3:landmarks)

    return slanted_pixel_arr, slanted_landmarks, projected_distance


def compute_zooming_region(slanted_landmarks: List[np.ndarray], W: int):
    slanted_ls = np.concatenate(slanted_landmarks, axis=0)
    min_xy = slanted_ls.min(axis=0)
    max_xy = slanted_ls.max(axis=0)
    center_xy = (min_xy + max_xy) / 2
    zoomed_width = int(1.5 * (max_xy - min_xy).max())
    min_xy = (center_xy - zoomed_width / 2).astype(int)
    max_xy = min_xy + zoomed_width
    min_xy = np.maximum(min_xy, 0)
    max_xy = np.minimum(max_xy, W)
    return min_xy, max_xy


def compute_Rodrigues_rotation_matrix(norm_vector: np.ndarray, theta: float):
    """Compute Rodrigues rotation matrix.
    Args:
        norm_vector (np.ndarray): norm vector (3,)
        theta (float): theta

    Returns:
        np.ndarray: rotation matrix (3,3)
    """
    norm_vector = norm_vector / np.linalg.norm(norm_vector)
    K = np.array([[0, -norm_vector[2], norm_vector[1]],
                  [norm_vector[2], 0, -norm_vector[0]],
                  [-norm_vector[1], norm_vector[0], 0]]) #(3,3)
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*K@K #(3,3)
    return R


def compute_error_of_landmark_displacement(logger: logtools.BaseLogger, landmarks: np.ndarray, predicted_landmarks: np.ndarray, csv_path: str, path_list: Union[List[str], np.ndarray], spacing: np.ndarray, patch_origin: Optional[np.ndarray], downsampling: int, result_dir: str, prename: str = "", postname: str = "") -> np.ndarray:
    """Compute error of landmark displacement.
    Args:
        logger (logtools.BaseLogger): logger
        landmarks (np.ndarray): landmarks at adjusted downsampled sp. (N,3,3)
        predicted_landmarks (np.ndarray): predicted landmarks at adjusted downsampled sp. (N,3,3)
        csv_path (str): csv path
        path_list (Union[List[str], np.ndarray]): path list (N,)
        spacing (np.ndarray): spacing (N,3)
        patch_origin (Optional[np.ndarray]): patch origin (N,3,3)
        downsampling (int): downsampling
        result_dir (str): result dir
        prename (str, optional): prename. Defaults to "".
        postname (str, optional): postname. Defaults to "".

    Returns:
        np.ndarray: distance error (N,)
    """
    # compute error of landmark displacement
    absolute_error = downsampling * np.abs(predicted_landmarks - landmarks) * np.absolute(spacing[:,None]) # (N,3,3) at adjusted sp.

    # load dataframe
    df = pd.read_csv(csv_path)
    if "Batch_dir" in df.columns:
        df["DICOMPath"] = df["Batch_dir"] + "/" + df["DICOMPath"]
    df = df.set_index("DICOMPath")
    df = df.loc[path_list]
    df_values = [np.array(path_list)[:,None], spacing, df["CTquality"].values[:,None]]
    df_columns = ["DICOMPath", "spacing_x", "spacing_y", "spacing_z", "CTquality"]

    distance_error = np.linalg.norm(absolute_error, axis=2) # (N,3) at original sp.
    df_values += [absolute_error.reshape(-1,9), distance_error] # (N,12)
    df_columns += [f"AE({i}-{j})" for i in ["RCC", "LCC", "NCC"] for j in ["x", "y", "z"]] + [f"DE({i})" for i in ["RCC", "LCC", "NCC"]]

    # compute distance between planes
    if patch_origin is None:
        patch_origin = 0
    distance_bw_planes_and_points, cosine_similarity, angle = compute_distance_between_planes(downsampling * spacing[:,None] * (landmarks + patch_origin), downsampling * spacing[:,None] * (predicted_landmarks + patch_origin)) # (2,N,3), (N,), (N,)
    horizontal_distance_error = np.sqrt(distance_error[None]**2 - distance_bw_planes_and_points**2) #(2,N,3)
    df_values += [distance_bw_planes_and_points.transpose(1,0,2).reshape(-1,6), horizontal_distance_error.transpose(1,0,2).reshape(-1,6)] # (N,12)
    df_columns += [f"Plane{pname}PDE({ln})" for pname in ["True", "Pred"] for ln in ["RCC", "LCC", "NCC"]] + [f"Plane{pname}HDE({ln})" for pname in ["True", "Pred"] for ln in ["RCC", "LCC", "NCC"]]

    # log errors
    if logger is not None:
        logger.log_metric(f"{prename} worst absolute error", np.max(absolute_error))
        for stat, stat_fn in zip(["mean", "median", "worst"], [np.mean, np.median, np.max]):
            logger.log_metric(f"{prename} {stat} distance error", stat_fn(distance_error))
            logger.log_metrics({f"{prename} {stat} {lname} distance error":s for lname, s in zip(["RCC", "LCC", "NCC"], stat_fn(distance_error, axis=0))})
            logger.log_metrics({f"{prename} {stat} {lname} projected distance error of {pname} plane":s for (pname, lname), s in zip(itertools.product(["true", "predicted"], ["RCC", "LCC", "NCC"]), stat_fn(distance_bw_planes_and_points, axis=1).reshape(-1))})
            logger.log_metrics({f"{prename} {stat} {lname} horizontal distance error of {pname} plane":s for (pname, lname), s in zip(itertools.product(["true", "predicted"], ["RCC", "LCC", "NCC"]), stat_fn(horizontal_distance_error, axis=1).reshape(-1))})
    
    mean_distance_bw_planes = distance_bw_planes_and_points.mean(axis=-1).mean(axis=0) #(N,)
    worst_distance_bw_planes = distance_bw_planes_and_points.max(axis=-1).max(axis=0) #(N,)
    projected_triangle_area_discrepancy = compute_triangle_area_discrepancy_from_original_to_projected(downsampling * spacing[:,None] * (landmarks + patch_origin), downsampling * spacing[:,None] * (predicted_landmarks + patch_origin)) #(N,)
    if logger is not None:
        for stat, stat_fn in zip(["mean", "median", "worst"], [np.mean, np.median, np.max]):
            logger.log_metric(f"{prename} {stat} mean distance between planes", stat_fn(mean_distance_bw_planes))
            logger.log_metric(f"{prename} {stat} worst distance between planes", stat_fn(worst_distance_bw_planes))
            logger.log_metric(f"{prename} {stat} cosine similarity", -stat_fn(-cosine_similarity))
            logger.log_metric(f"{prename} {stat} angle", stat_fn(angle))
            logger.log_metric(f"{prename} {stat} projected triangle area discrepancy", stat_fn(projected_triangle_area_discrepancy))
    df_values += [mean_distance_bw_planes[:,None], worst_distance_bw_planes[:,None], cosine_similarity[:,None], angle[:,None], projected_triangle_area_discrepancy[:,None]] # (N,3)
    df_columns += ["MeanDistPlanes", "WorstDistPlanes", "CosSim", "Angle", "TAD"]

    # log df
    df_values = np.concatenate(df_values, axis=1) # (N,29)
    df = pd.DataFrame(df_values, columns=df_columns)
    if logger is not None:
        logger.log_table(f"{prename} samplewise error of landmark displacement.csv", df)
    df.to_csv(os.path.join(result_dir, f"{prename}_samplewise_error{postname}.csv"), index=False)
    return distance_error.max(1) #(N,)


def record_cv_errors(logger: logtools.BaseLogger, result_dir: str, n_splits: int = 5):
    """Record cv errors.
    Args:
        logger (logtools.BaseLogger): logger
        result_dir (str): result dir
        n_splits (int, optional): n_splits. Defaults to 5.
    """
    columns = [f"DE({i})" for i in ["RCC", "LCC", "NCC"]] + ["MeanDistPlanes", "WorstDistPlanes", "CosSim", "Angle"]
    stat_names = ["mean", "median", "worst"]
    record_dict = {pre+k:[] for pre in ["", "test-"] for k in [i+j for i in stat_names for j in columns + ["DE"]]}

    for phase in ["valid", "test"]:
        pre = "" if phase=="valid" else phase+"-"
        for i in range(n_splits):
            df = pd.read_csv(os.path.join(result_dir, f"{phase}_samplewise_error_cv{i}.csv"))
            for stat, stat_fn in zip(stat_names, [np.mean, np.median, np.max]):
                for c in columns:
                    if stat=="worst" and c=="CosSim":
                        record_dict[pre+stat+c].append(-stat_fn(-df[c]))
                    else:
                        record_dict[pre+stat+c].append(stat_fn(df[c]))
                record_dict[pre+stat+"DE"].append(stat_fn(df[columns[:3]].values.flatten()))

    # mean aggregation of record_dict
    record_dict = {k:np.mean(v) for k,v in record_dict.items()}
    logger.log_metrics(record_dict)
    
    # save record_dict
    with open(os.path.join(result_dir, "cv_errors.json"), "w") as f:
        json.dump(record_dict, f)

    return record_dict


def record_ensemble_prediction(logger: logtools.BaseLogger, result_dir: str, n_splits: int = 5, parent_dir: Optional[str] = None):
    """Record ensemble prediction.
    Args:
        logger (logtools.BaseLogger): logger
        result_dir (str): result dir
        n_splits (int, optional): n_splits. Defaults to 5.
        parent_dir (Optional[str], optional): parent dir. Defaults to None.
    """
    if parent_dir is None:
        parent_dir = result_dir
    df = pd.read_csv(os.path.join(parent_dir, "cross_validation_split.csv"))
    df = df.merge(pd.read_csv(os.path.join(result_dir, "predicted_landmark_cv0.csv")).rename(columns={f"p{ln}_{coord}":f"p{ln}_{coord}_cv0" for ln in ["RCC", "LCC", "NCC"] for coord in ["x", "y", "z"]}), on="DICOMPath")
    for i in range(1, n_splits):
        cvdf = pd.read_csv(os.path.join(result_dir, f"predicted_landmark_cv{i}.csv")).rename(columns={f"p{ln}_{coord}":f"p{ln}_{coord}_cv{i}" for ln in ["RCC", "LCC", "NCC"] for coord in ["x", "y", "z"]})
        cvdf = cvdf[["DICOMPath"] + [f"p{ln}_{coord}_cv{i}" for ln in ["RCC", "LCC", "NCC"] for coord in ["x", "y", "z"]]]
        df = df.merge(cvdf, on="DICOMPath")

    # extract test (fold < 0) data
    df = df[df["fold"] < 0]
    
    # ensemble prediction
    for ln in ["RCC", "LCC", "NCC"]:
        for coord in ["x", "y", "z"]:
            df[f"p{ln}_{coord}"] = df[[f"p{ln}_{coord}_cv{i}" for i in range(n_splits)]].mean(axis=1)

    # extract landmark points 3x3
    landmarks = df[[f"{ln}_{coord}" for ln in ["RCC", "LCC", "NCC"] for coord in ["x", "y", "z"]]].values.reshape(-1,3,3)
    predicted_landmarks = df[[f"p{ln}_{coord}" for ln in ["RCC", "LCC", "NCC"] for coord in ["x", "y", "z"]]].values.reshape(-1,3,3)

    # compute distance b/w true and predicted landmarks
    distance_error = np.linalg.norm(predicted_landmarks - landmarks, axis=2) #(N,nl:3)
    df[[f"DE({ln})" for ln in ["RCC", "LCC", "NCC"]]] = distance_error

    # compute distance between planes
    distance_bw_planes_and_points, cosine_similarity, angle = compute_distance_between_planes(landmarks, predicted_landmarks) #(np:2,N,nl:3),(N,),(N,)
    horizontal_distance_error = np.sqrt(distance_error[None]**2 - distance_bw_planes_and_points**2) #(np:2,N,nl:3)
    mean_distance_bw_planes = distance_bw_planes_and_points.mean(axis=-1).mean(axis=0) #(N,)
    worst_distance_bw_planes = distance_bw_planes_and_points.max(axis=-1).max(axis=0) #(N,)
    df[[f"Plane{pname}PDE({ln})" for pname in ["True", "Pred"] for ln in ["RCC", "LCC", "NCC"]]] = distance_bw_planes_and_points.transpose(1,0,2).reshape(-1,6) # (N,6) projected distance error
    df[[f"Plane{pname}HDE({ln})" for pname in ["True", "Pred"] for ln in ["RCC", "LCC", "NCC"]]] = horizontal_distance_error.transpose(1,0,2).reshape(-1,6) # (N,6) horizontal distance error
    projected_triangle_area_discrepancy = compute_triangle_area_discrepancy_from_original_to_projected(landmarks, predicted_landmarks) #(N,)
    df["MeanDistPlanes"] = mean_distance_bw_planes
    df["WorstDistPlanes"] = worst_distance_bw_planes
    df["CosSim"] = cosine_similarity
    df["Angle"] = angle
    df["TAD"] = projected_triangle_area_discrepancy

    # save ensemble prediction
    df.to_csv(os.path.join(result_dir, "predicted_landmark.csv"), index=False)

    # log statistics of ensemble prediction
    stat_names = ["mean", "median", "worst"]
    for stat, stat_fn in zip(stat_names, [np.mean, np.median, np.max]):
        for ln in ["RCC", "LCC", "NCC"]:
            logger.log_metric(f"ep-{stat}DE({ln})", stat_fn(distance_error[:,["RCC", "LCC", "NCC"].index(ln)]))
        logger.log_metric(f"ep-{stat}DE", stat_fn(distance_error))
        logger.log_metric(f"ep-{stat}MeanDistPlanes", stat_fn(mean_distance_bw_planes))
        logger.log_metric(f"ep-{stat}WorstDistPlanes", stat_fn(worst_distance_bw_planes))
        logger.log_metric(f"ep-{stat}CosSim", -stat_fn(-cosine_similarity))
        logger.log_metric(f"ep-{stat}Angle", stat_fn(angle))
        logger.log_metric(f"ep-{stat}TAD", stat_fn(projected_triangle_area_discrepancy))


def plot_on_slanted_plane_for_ensemble_prediction(logger: logtools.BaseLogger, csv_path:str, result_dir: str, target_spacing: Optional[float] = None, expanding_coefficient: float = 1.5, voxel_shapes: List[int] = [512,512,320]):
    """Plot on slanted plane for ensemble prediction.
    Args:
        logger (logtools.BaseLogger): logger
        csv_path (str): csv path
        result_dir (str): result dir
        target_spacing (Optional[float], optional): target spacing. Defaults to None.
        voxel_shapes (List[int], optional): voxel shapes. Defaults to [512,512,320].
    """
    df = pd.read_csv(os.path.join(result_dir, "predicted_landmark.csv"))
    landmarks = df[[f"{ln}_{coord}" for ln in ["RCC", "LCC", "NCC"] for coord in ["x", "y", "z"]]].values.astype("float32").reshape(-1,3,3) #(N,nl:3,nc:3)
    predicted_landmarks = df[[f"p{ln}_{coord}" for ln in ["RCC", "LCC", "NCC"] for coord in ["x", "y", "z"]]].values.astype("float32").reshape(-1,3,3) #(N,nl:3,nc:3)
    spacing = df[["spacing_x", "spacing_y", "spacing_z"]].values.astype("float32") #(N,3)
    origin = df[["origin_x", "origin_y", "origin_z"]].values.astype("float32") #(N,3)
    path_list = df["DICOMPath"].values #(N,)
    arr_path_list = [os.path.join(csv_path.rsplit("/", 1)[0], path, "voxel_array.npy") for path in path_list]

    # modify target spacing
    original_target_spacing = target_spacing
    target_spacing = spacing if target_spacing is None else target_spacing * np.sign(spacing) # signed target spacing (N,3)
    adjusted_voxel_shapes = np.ceil(np.array(voxel_shapes) * spacing / target_spacing).astype(int) #(N,3)
    target_spacing = spacing * np.array(voxel_shapes) / adjusted_voxel_shapes #(N,3)

    # compute landmarks at adjusted grid sp.
    landmarks = data.transform_landmarks_to_downsampled_voxel(landmarks, target_spacing, origin) #(N,3,3) at adjusted signed grid sp.
    predicted_landmarks = data.transform_landmarks_to_downsampled_voxel(predicted_landmarks, target_spacing, origin) #(N,3,3) at adjusted signed grid sp.

    # compute patch origin for plotting on slanted plane
    min_xyz = landmarks.min(axis=1) #(N,nc:3)
    max_xyz = landmarks.max(axis=1) #(N,nc:3)
    half_patch_size = (expanding_coefficient * (max_xyz - min_xyz).max(axis=1)).astype(int) #(N,)
    patch_origin = np.maximum((min_xyz - half_patch_size[:,None] // 2).astype(int), 0) #(N,nc:3)
    patch_size = 2 * half_patch_size #(N,)
    patch_terminal = patch_origin + patch_size[:,None] #(N,nc:3)

    # compute patched landmarks
    landmarks -= patch_origin[:,None] #(N,nc:3,nc:3)
    predicted_landmarks -= patch_origin[:,None] #(N,nc:3,nc:3)
    
    # load voxel array
    total_obs = []
    for po, pt, s, ts, avs, p in zip(patch_origin, patch_terminal, spacing, target_spacing, adjusted_voxel_shapes, arr_path_list):
        if os.path.exists(p.replace(".npy", f"_sp{original_target_spacing}.npy")):
            arr = np.load(p.replace(".npy", f"_sp{original_target_spacing}.npy")).astype("float32") #(W,H,D)
        else:
            arr = np.load(p).astype("float32").swapaxes(0,1) #(W,H,D)
            if np.any(s != ts):
                arr = data.adjust_spacing_of_voxel_array_for_patch(arr, s, ts, avs) #(aW,aH,aD)
        total_obs.append(arr[po[0]:pt[0], po[1]:pt[1], po[2]:pt[2]][None]) #(1,pW,pH,pD)

    # plot on slanted plane
    plot_landmarks_at_slanted_pixels(total_obs, predicted_landmarks, path_list, target_spacing, patch_size / expanding_coefficient, landmarks, logger, result_dir, False, "ep")



def compute_distance_between_planes(landmarks: np.ndarray, predicted_landmarks: np.ndarray) -> np.ndarray:
    """
    Args:
        landmarks (np.ndarray): landmarks at original sp. (N,3,3)
        predicted_landmarks (np.ndarray): predicted landmark at original sp. (N,3,3)

    Returns:
        np.ndarray: distance between planes (2,N,3)
        np.ndarray: cosine similarity (N,)
        np.ndarray: absolute angle between two planes (N,)
    """
    landmarks = np.stack([landmarks, predicted_landmarks]) #(2,N,3:landmarks,3:xyz)
    plane_params = np.linalg.pinv(landmarks).sum(axis=-1) #(2,N,3:xyz)
    distance = np.absolute((plane_params[:,:,None] * np.flip(landmarks, 0)).sum(-1) - 1) / np.linalg.norm(plane_params[:,:,None], axis=-1) #(2,N,3)
    norm_vector = plane_params / np.linalg.norm(plane_params, axis=-1, keepdims=True) #(2,N,3:xyz)
    # compute cosine similarity
    cosine_similarity = (norm_vector[0] * norm_vector[1]).sum(axis=-1) #(N,)
    # compute absolute angle between two planes
    angle = np.degrees(np.absolute(np.arccos(cosine_similarity))) #(N,)
    return distance, cosine_similarity, angle


def compute_triangle_area_discrepancy_from_original_to_projected(landmarks: np.ndarray, predicted_landmarks: np.ndarray):
    """
    Args:
        landmarks (np.ndarray): landmarks at original sp. (N,nl:3,coord:3)
        predicted_landmarks (np.ndarray): predicted landmark at original sp. (N,nl:3,coord:3)

    Returns:
        np.ndarray: triangle area discrepancy (N,)
    """
    landmarks = np.stack([landmarks, predicted_landmarks]) #(2,N,3:landmarks,3:xyz)
    surface_area = 0.5 * np.linalg.norm(np.cross(landmarks[:,:,1] - landmarks[:,:,0], landmarks[:,:,2] - landmarks[:,:,0]), axis=-1) #(2,N)

    # compute normal vector of planes
    plane_params = np.linalg.pinv(landmarks).sum(axis=-1) #(2,N,3:xyz)
    norm_vector = plane_params / np.linalg.norm(plane_params, axis=-1, keepdims=True) #(2,N,3:xyz)
    norm_vector = np.flip(norm_vector, axis=0) #(2,N,3:xyz)

    # projected landmarks
    projected_landmarks = landmarks - ((landmarks - np.flip(landmarks, axis=0)[:,:,0][:,:,None]) * norm_vector[:,:,None]).sum(axis=-1)[...,None] * norm_vector[:,:,None] #(2,N,3:landmarks,3:xyz)
    projected_surface_area = 0.5 * np.linalg.norm(np.cross(projected_landmarks[:,:,1] - projected_landmarks[:,:,0], projected_landmarks[:,:,2] - projected_landmarks[:,:,0]), axis=-1) #(2,N)

    # compute surface area discrepancy
    projected_triangle_area_discrepancy = np.abs(surface_area - projected_surface_area).mean(axis=0) #(N,)
    return projected_triangle_area_discrepancy



def compute_norm_vector(landmarks: np.ndarray):
    """
    Args:
        landmarks (np.ndarray): landmarks at original sp. (N,3,3)

    Returns:
        np.ndarray: norm vector (N,3)
    """
    plane_params = np.linalg.pinv(landmarks).sum(axis=-1) #(N,3)
    norm_vector = plane_params / np.linalg.norm(plane_params, axis=-1, keepdims=True) #(N,3)
    return norm_vector


def compute_predicted_landmark_at_original_space(predicted_landmarks: np.ndarray, spacing: np.ndarray, origin: np.ndarray, patch_origin: Optional[np.ndarray], downsampling: int):
    """Save predicted landmark at original sp.
    Args:
        predicted_landmarks (np.ndarray): weighted landmark displacement at adjusted downsampled sp. (N,3,3)
        spacing (np.ndarray): spacing (N,3)
        origin (np.ndarray): origin (N,3)
        patch_origin (Optional[np.ndarray]): patch origin (N,3,3)
        downsampling (int): downsampling
        result_dir (str): result dir

    Returns:
        np.ndarray: predicted landmark at original sp. (N,3,3)
    """
    # save predicted landmark
    if patch_origin is None:
        patch_origin = 0
    predicted_landmark = origin[:,None] + downsampling * spacing[:,None] * (predicted_landmarks + patch_origin) # (N,3,3) at original sp.
    return predicted_landmark


def save_predicted_landmark(predicted_landmark: np.ndarray, csv_path: str, path_list: Union[List[str], np.ndarray], result_dir: str, postname: str = ""):
    """Save predicted landmark.
    Args:
        predicted_landmark (np.ndarray): predicted landmark at original sp. (N,3,3)
        csv_path (str): csv path
        path_list (Union[List[str], np.ndarray]): path list (N,)
        result_dir (str): result dir
        postname (str, optional): postname. Defaults to "".
    """
    df = pd.read_csv(csv_path)
    if "Batch_dir" in df.columns:
        df["DICOMPath"] = df["Batch_dir"] + "/" + df["DICOMPath"]
    df = df.set_index("DICOMPath", drop=False)
    df = df.loc[path_list]
    df[["pRCC_x", "pRCC_y", "pRCC_z", "pLCC_x", "pLCC_y", "pLCC_z", "pNCC_x", "pNCC_y", "pNCC_z"]] = predicted_landmark.reshape(-1,9)
    df.to_csv(os.path.join(result_dir, f"predicted_landmark{postname}.csv"), index=False)



def boxplot_by_quality(result_dir: str, logger: logtools.BaseLogger, phase: str, unifying_csv_on: bool = False, figsize: tuple = (8,4)):
    """Boxplot by quality.
    Args:
        result_dir (str): result dir
        logger (logtools.BaseLogger): logger
        phase (str): phase
        figsize (tuple, optional): figsize. Defaults to (8,4).
    """
    if unifying_csv_on:
        df = pd.read_csv(os.path.join(result_dir, "predicted_landmark.csv"))
    else:
        df = pd.read_csv(os.path.join(result_dir, f"{phase}_samplewise_error.csv"))
    partial_df = df[["CTquality", "DE(RCC)", "DE(LCC)", "DE(NCC)", "WorstDistPlanes"]]
    partial_df.columns = ["CTquality", "RCC", "LCC", "NCC", "Plane"]
    melt_df = pd.melt(partial_df, id_vars="CTquality", var_name="Landmark", value_name=f"Euclid error (mm)")

    # plot
    fig, ax = plt.subplots(1,1,figsize=figsize)
    sns.boxplot(x="Landmark", y=f"Euclid error (mm)", data=melt_df, hue="CTquality", ax=ax)
    fig.tight_layout()

    # existing results by Noothout+2020
    for i, v in enumerate([2.23, 2.40, 2.48]):
        ax.axhline(v, xmin=i/3, xmax=(i+1)/3, ls="--", c="k", label="RW")

    # save
    if unifying_csv_on:
        fig.savefig(os.path.join(result_dir, f"boxplot_by_quality.pdf"), bbox_inches="tight")
        logger.log_figure(f"boxplot by quality", fig)
    else:
        fig.savefig(os.path.join(result_dir, f"{phase}_boxplot_by_quality.pdf"), bbox_inches="tight")
        logger.log_figure(f"{phase} boxplot by quality", fig)
