import os, argparse
from typing import Optional
import numpy as np
import torch
import loss, data, logtools


def train(m: torch.nn.Module, args: argparse.Namespace, device: torch.device, logger: logtools.BaseLogger, train_input: np.ndarray, valid_input: np.ndarray, train_output: np.ndarray, valid_output: np.ndarray, class_weights: Optional[np.ndarray] = None, post_name: str=""):
    """Train model.
    Args:
        m (torch.nn.Module): model
        args (argparse.Namespace): arguments
        device (torch.device): device
        logger (logtools.BaseLogger): logger
        train_input (np.ndarray): train input (N,1,W,H,D)
        train_output (np.ndarray): train output (N,3,W,H,D)
        valid_input (np.ndarray): valid input (N,1,W,H,D)
        valid_output (np.ndarray): valid output (N,3,W,H,D)
        class_weights (Optional[np.ndarray], optional): class weights. Defaults to None.
        post_name (str, optional): post name. Defaults to "".
    """
    balancing_weight_on = args.loss_function == "WBCE"

    # set optimizer
    optimizer = getattr(torch.optim, args.optimizer)(m.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.decay_factor, patience=args.patience)

    # set loss function
    criterion = loss.get_loss_fn(args.loss_function, class_weights, args.focal_loss_alpha, args.focal_loss_gamma, args.gradient_penalty)
    if args.use_image_lipschitz_loss:
        criterion = [criterion, loss.ImageLipschitzLoss(args.gradient_penalty)]

    # set dataloader
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_input).float(), torch.from_numpy(train_output).float())
    valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(valid_input).float(), torch.from_numpy(valid_output).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # train
    optimizer.zero_grad()
    current_every_step = 1
    for epoch in range(args.n_epochs):
        m.train()
        train_loss = 0
        for x, y in train_loader:
            if args.patch_perturbation > 0 and args.n_model_steps > 1:
                x, y = data.compute_perturbed_voxel_arr_and_heatmaps(x, y, args.patch_size, args.heatmap_type) #(N,3,W,H,D), (N,3,W,H,D)
            x, y = x.to(device), y.to(device)
            y_hat = m(x)
            if type(criterion) == list:
                _loss = sum([c(y_hat, y) for c in criterion])
            else:
                _loss = criterion(y_hat, y)
            _loss.backward()
            train_loss += _loss.item()

            current_every_step += 1
            if current_every_step%args.n_every_steps==0:
                optimizer.step()
                optimizer.zero_grad()
                current_every_step = 1
            
        m.eval()
        valid_loss = 0
        valid_diff = np.zeros(3)
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            y_hat = m(x)
            if balancing_weight_on:
                _loss = (torch.exp(y_hat) * criterion(y_hat, y)).mean()
            else:
                if type(criterion) == list:
                    _loss = sum([c(y_hat, y) for c in criterion])
                else:
                    _loss = criterion(y_hat, y)
            valid_loss += _loss.item()

        train_loss /= len(train_loader)
        if len(valid_loader) > 0:
            valid_loss /= len(valid_loader)
            valid_diff /= len(valid_loader)
        else:
            valid_loss = train_loss
        scheduler.step(valid_loss)

        # log loss
        logger.log_metrics({f"train_loss{post_name}": train_loss, 
                            f"valid_loss{post_name}": valid_loss}, epoch=epoch)

        print(f"Epoch[{epoch}/{args.n_epochs}] train loss:{train_loss:.4f}, valid loss:{valid_loss:.4f}")

    # save model
    torch.save(m.state_dict(), os.path.join(args.result_dir, f"state_dict{post_name}.pth"))
    