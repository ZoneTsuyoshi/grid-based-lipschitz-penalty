import os, argparse, time, itertools, copy
import numpy as np
import torch
import data, train, predict, models, logtools, utils, parse


def main_ho(args: argparse.Namespace, device: torch.device, logger: logtools.BaseLogger, global_phase: bool):
    start_time = time.time()

    # load data
    if global_phase:
        voxel_arr, heatmaps, spacing, origin, weights, landmarks, total_path_list = data.load_data(args.data_path, args.downsampling, args.handling_samples_over_max_length, args.max_z_length, args.spacing, args.heatmap_sigma, args.gaussian_height, args.loss_function in ["WCE", "WBCE"], args.debug)
        patch_origin = None
        predicted_landmarks = None

        # set model
        m = getattr(models, args.model_name)(1, 3, 1, args.hidden_dim, args.layer_order, args.num_groups_groupnorm, args.n_layers, args.padding, True, False)

        # split data
        voxel_arr_list, heatmaps_list, spacing_list, origin_list, patch_origin_list, landmarks_list, path_list_list, idx_list = data.train_valid_test_split([voxel_arr, heatmaps, spacing, origin, patch_origin, landmarks, total_path_list], args.valid_ratio, args.test_ratio, args.split_seed)
    else:
        predicted_landmarks, total_path_list = data.load_predicted_results(args.parent_dir, args.debug) #(N,3,3), (N,) at original sp.

        # split data
        path_list_list, idx_list = data.train_valid_test_split([total_path_list], args.valid_ratio, args.test_ratio, args.split_seed)

        # load patched data
        voxel_arr_list, landmarks_list, spacing_list, origin_list, patch_origin_list, heatmaps_list = [list(itertools.repeat(None, 3)) for _ in range(6)]
        for i, path_list, idx, patch_size in zip(range(3), path_list_list, idx_list, [args.patch_size + 2*args.patch_perturbation, args.patch_size, args.patch_size]):
            voxel_arr_list[i], landmarks_list[i], spacing_list[i], origin_list[i], patch_origin_list[i], heatmaps_list[i] = data.get_patched_voxel_array_and_heatmaps(args.data_path, path_list, predicted_landmarks[idx], patch_size, args.spacing, args.downsampling, args.heatmap_sigma) #(N,3,W+2pp,H+2pp,D+2pp), (N,3,3), (N,3), (N,3), (N,3,3), (N,3,W+2pp,H+2pp,D+2pp)

        weights = data.compute_weights(heatmaps_list[0], args.loss_function in ["WCE", "WBCE"], args.heatmap_type)
        m = getattr(models, args.model_name)(3, 3, 3, args.hidden_dim, args.layer_order, args.num_groups_groupnorm, args.n_layers, args.padding, True, False)

    print("Data loading time: {:.3f} sec".format(time.time() - start_time))

    # train
    print(m)
    if args.test:
        m.load_state_dict(torch.load(os.path.join(args.result_dir, "hr_state_dict.pth")))
        m.to(device)
    else:
        start_time = time.time()
        logger.set_model_graph(m)
        m.to(device)
        train.train(m, args, device, logger, *voxel_arr_list[:2], *heatmaps_list[:2], weights, heatmaps_list[1].shape[2:])
        print("Training time: {:.3f} sec".format(time.time() - start_time))

    # predict
    start_time = time.time()
    predict.generate_results(args, device, m, logger, global_phase, total_path_list, voxel_arr_list, heatmaps_list, spacing_list, origin_list, patch_origin_list, landmarks_list, path_list_list, predicted_landmarks, idx_list)
    print("Prediction time: {:.3f} sec".format(time.time() - start_time))



def main_cv(args: argparse.Namespace, device: torch.device, logger: logtools.BaseLogger, global_phase: bool):

    # load data
    if not args.log:
        if global_phase:
            start_time = time.time()
            voxel_arr, heatmaps, spacing, origin, weights, landmarks, total_path_list = data.load_data(args.data_path, args.downsampling, args.handling_samples_over_max_length, args.max_z_length, args.spacing, args.heatmap_sigma, args.gaussian_height, args.loss_function in ["WCE", "WBCE"], args.debug)
            print("Data loading time: {:.3f} sec".format(time.time() - start_time))

            # split data
            idx_list, test_idx = data.cross_validation_split(total_path_list, args.result_dir, args.cross_validation, args.test_ratio, args.split_seed)
            path_list = copy.deepcopy(total_path_list)

            for i in range(args.cross_validation):
                if not args.test and os.path.exists(os.path.join(args.result_dir, f"hr_state_dict{i}.pth")):
                    continue
                
                print(f"============= fold {i} ==============")

                # set model
                m = getattr(models, args.model_name)(1, 3, 1, args.hidden_dim, args.layer_order, args.num_groups_groupnorm, args.n_layers, args.padding, True, False)

                # set train and valid data
                valid_idx = idx_list[i]
                train_idx = np.delete(np.arange(len(voxel_arr)), np.concatenate([valid_idx, test_idx]))
                voxel_arr_list, heatmaps_list, spacing_list, origin_list, landmarks_list, path_list_list = [[v[train_idx], v[valid_idx], v[test_idx]] for v in [voxel_arr, heatmaps, spacing, origin, landmarks, path_list]]

                # train
                print(m)
                if args.test:
                    m.load_state_dict(torch.load(os.path.join(args.result_dir, f"hr_state_dict{i}.pth")))
                    m.to(device)
                else:
                    start_time = time.time()
                    logger.set_model_graph(m)
                    m.to(device)
                    train.train(m, args, device, logger, *voxel_arr_list[:2], *heatmaps_list[:2], weights, heatmaps_list[0].shape[2:], str(i))
                    print("Training time: {:.3f} sec".format(time.time() - start_time))

                # predict
                start_time = time.time()
                predict.generate_results(args, device, m, None, global_phase, total_path_list, voxel_arr_list, heatmaps_list, spacing_list, origin_list, itertools.repeat(None, 3), landmarks_list, path_list_list, None, [train_idx, valid_idx, test_idx], ["train", "valid", "test"], False, f"_cv{i}")
                print("Prediction time: {:.3f} sec".format(time.time() - start_time))
        else:
            split_df, total_path_list = data.load_split_dataframe(args.parent_dir, args.debug)
            test_idx = split_df[split_df["fold"] < 0].index.values
            split_df = split_df[split_df["fold"] >= 0]
            for i in range(args.cross_validation):
                print(f"============= fold {i} ==============")
                predicted_landmarks, _ = data.load_predicted_results(args.parent_dir, args.debug, f"_cv{i}", total_path_list) #(N,3,3), (N,) at original sp.
                train_idx = split_df[split_df["fold"] != i].index.values
                valid_idx = split_df[split_df["fold"] == i].index.values
                idx_list = [train_idx, valid_idx, test_idx]
                path_list_list = [total_path_list[j] for j in idx_list]

                # load patched data
                voxel_arr_list, landmarks_list, spacing_list, origin_list, patch_origin_list, heatmaps_list = [list(itertools.repeat(None, 3)) for _ in range(6)]
                for j, path_list, idx, patch_size, pred_flag in zip(range(3), path_list_list, idx_list, [args.patch_size + 2*args.patch_perturbation, args.patch_size, args.patch_size], [False, True, True]):
                    voxel_arr_list[j], landmarks_list[j], spacing_list[j], origin_list[j], patch_origin_list[j], heatmaps_list[j] = data.get_patched_voxel_array_and_heatmaps(args.data_path, path_list, predicted_landmarks[idx], patch_size, args.spacing, args.downsampling, args.heatmap_sigma, pred_flag) #(N,3,W+2pp,H+2pp,D+2pp), (N,3,3), (N,3), (N,3), (N,3,3), (N,3,W+2pp,H+2pp,D+2pp)

                weights = data.compute_weights(heatmaps_list[0], args.loss_function in ["WCE", "WBCE"], args.heatmap_type)
                m = getattr(models, args.model_name)(3, 3, 3, args.hidden_dim, args.layer_order, args.num_groups_groupnorm, args.n_layers, args.padding, True, False)

                # train
                print(m)
                if args.test:
                    m.load_state_dict(torch.load(os.path.join(args.result_dir, f"hr_state_dict{i}.pth")))
                    m.to(device)
                else:
                    start_time = time.time()
                    logger.set_model_graph(m)
                    m.to(device)
                    train.train(m, args, device, logger, *voxel_arr_list[:2], *heatmaps_list[:2], weights, heatmaps_list[0].shape[2:], str(i))
                    print("Training time: {:.3f} sec".format(time.time() - start_time))

                # predict
                start_time = time.time()
                predict.generate_results(args, device, m, None, global_phase, total_path_list, voxel_arr_list, heatmaps_list, spacing_list, origin_list, patch_origin_list, landmarks_list, path_list_list, predicted_landmarks, [train_idx, valid_idx, test_idx], ["train", "valid", "test"], False, f"_cv{i}")
                print("Prediction time: {:.3f} sec".format(time.time() - start_time))

    # record cv errors
    predict.record_cv_errors(logger, args.result_dir, args.cross_validation)
    predict.record_ensemble_prediction(logger, args.result_dir, args.cross_validation, args.parent_dir)
    predict.boxplot_by_quality(args.result_dir, logger, None, True)
    predict.plot_on_slanted_plane_for_ensemble_prediction(logger, args.data_path, args.result_dir, args.spacing)


def main(args: argparse.Namespace):
    postname = "" if args.cross_validation <= 1 else f'_cv{args.cross_validation}'
    args, device, logger, global_phase = utils.initialize(args, f"GLiP-{postname}")

    if args.cross_validation > 1:
        main_cv(args, device, logger, global_phase)
    else:
        main_ho(args, device, logger, global_phase)

    
if __name__ == "__main__":
    parser = parse.get_parser()
    args = parser.parse_args()
    postname = "" if args.cross_validation <= 1 else f'_cv{args.cross_validation}'
    args = parse.convert_args_whether_test(args, f"Tan19_{args.regression_type}{postname}")
    args = parse.convert_args_misc(args)
    main(args)