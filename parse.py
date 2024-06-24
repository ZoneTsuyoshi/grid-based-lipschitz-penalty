import argparse, datetime, os, shutil, json
from distutils.util import strtobool
from typing import Optional


def get_common_parser(description: str = "TAVI"):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # basic
    parser.add_argument("-d", "--debug", action="store_true", help="debug")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("-pd", "--parent_dir", type=str, default=None, help="parent directory name for storing results")
    parser.add_argument("-rd", "--result_dir", type=str, default=None, help="directory name for storing results")
    parser.add_argument("-ek", "--experiment_key", type=str, default=None, help="experiment key for storing results")
    parser.add_argument("-nms", "--n_model_steps", type=int, default=1, help="number of model steps")
    parser.add_argument("-t", "--test", action="store_true", help="test mode")
    parser.add_argument("-l", "--log", action="store_true", help="log mode")

    # data setting
    parser.add_argument("-dp", "--data_path", type=str, default="data/landmarks.csv", help="data path")
    parser.add_argument("-ds", "--downsampling", type=int, default=4, help="downsampling")
    parser.add_argument("-hsoml", "--handling_samples_over_max_length", type=str, default="del", help="handling samples over max length")
    parser.add_argument("-mzl", "--max_z_length", type=int, default=320, help="max z length")
    parser.add_argument("-sp", "--spacing", type=float_or_none, default=None, help="spacing")
    parser.add_argument("-hs", "--heatmap_sigma", type=float, default=1.0, help="heatmap sigma")
    parser.add_argument("-ps", "--patch_size", type=int, default=64, help="local patch size")
    parser.add_argument("-pp", "--patch_perturbation", type=int, default=0, help="patch perturbation")
    parser.add_argument("-se", "--seed", type=int, default=0, help="seed")
    parser.add_argument("-ss", "--split_seed", type=int, default=0, help="split seed")

    # training setting
    parser.add_argument("-e", "--n_epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("-nes", "--n_every_steps", type=int, default=1, help="number of every steps")
    parser.add_argument("-bs", "--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("-opt", "--optimizer", type=str, default="Adam", help="optimizer")
    parser.add_argument("-sch", "--scheduler", type=str, default="ReduceLROnPlateau", help="scheduler")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose printing")
    parser.add_argument("-pe", "--print_every", type=int, default=1, help="every epoch for printing results")
    parser.add_argument("-df", "--decay_factor", type=float, default=0.1, help="decay ratio of scheduler")
    parser.add_argument("-pat", "--patience", type=int, default=10, help="decay patience of scheduler")
    parser.add_argument("-pei", "--print_every_iteration", action="store_true", help="print every iteration")
    parser.add_argument("-vr", "--valid_ratio", type=float, default=0.1, help="validation ratio")
    parser.add_argument("-tr", "--test_ratio", type=float, default=0.1, help="test ratio")
    parser.add_argument("-cv", "--cross_validation", type=int, default=0, help="cross validation")

    # logging setting
    parser.add_argument("-log", "--logger", type=str, choices=["Empty", "EmptyLogger", "Comet", "CometLogger", "Neptune", "NeptuneLogger"], default="CometLogger", help="logger")
    parser.add_argument("-nolog", "--no_logger", action="store_true", help="no logger")
    parser.add_argument("-me", "--monitor_every", type=int, default=10, help="every epoch for monitoring metrics")
    parser.add_argument("-ntp", "--number_of_training_plots", type=int, default=5, help="number of training plots")
    
    # model setting
    parser.add_argument("-nl", "--n_layers", type=int, default=4, help="number of depth layers")
    parser.add_argument("-uill", "--use_image_lipschitz_loss", type=strtobool, default="false", help="use image lipschitz loss")
    parser.add_argument("-mn", "--model_name", type=str, default="UNet3D", choices=["UNet3D", "ResidualUNet3D", "ResidualUNetSE3D"], help="model")
    parser.add_argument("-hd", "--hidden_dim", type=int, default=16, help="number of hidden dim.")
    parser.add_argument("-ngg", "--num_groups_groupnorm", type=int, default=1, help="number of groups for GroupNorm")
    parser.add_argument("-ngc", "--num_groups_conv", type=int, default=1, help="number of groups for Conv layer")
    parser.add_argument("-lo", "--layer_order", type=str, default="bcr", help="layer order")
    parser.add_argument("-pad", "--padding", type=int, default=1, help="padding")

    # loss setting
    parser.add_argument("-lf", "--loss_function", type=str, default="GLiP", choices=["FL", "CE", "WCE", "MSE", "L1", "SL1", "GLiP", "NGLiP"], help="loss function")
    parser.add_argument("-fla", "--focal_loss_alpha", type=float, default=0.8, help="focal loss alpha")
    parser.add_argument("-flg", "--focal_loss_gamma", type=int, default=2, help="focal loss gamma")
    parser.add_argument("-gp", "--gradient_penalty", type=float, default=1., help="gradient penalty")

    return parser



def get_result_direcotry_name(exp_name: str, debug: bool = False, result_dir: Optional[str] = None):
    if result_dir is None:
        dt_now = datetime.datetime.now()
        upper_dir = "results"

        if debug:
            name = "d"
            if os.path.exists(os.path.join(upper_dir, "d")):
                shutil.rmtree(os.path.join(upper_dir, "d"))
        else:
            name = "{}_{}_".format(dt_now.strftime("%y%m%d"), exp_name)
            i = 1
            while os.path.exists(os.path.join(upper_dir, name + str(i))):
                i += 1
            name += str(i)
            
        print(name)
        result_dir = os.path.join(upper_dir, name)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    return result_dir


def convert_args_whether_test(args: argparse.Namespace, exp_name: str) -> argparse.Namespace:
    """Convert args whether test or not."""
    if args.test or args.log:
        log = args.log
        config = vars(args)
        with open(os.path.join(args.result_dir, "config.json"), "r") as f:
            config.update(json.load(f))
        args = argparse.Namespace(**config)
        if log:
            args.log = True
        args.test = True
    elif args.one_more_trying:
        config = vars(args)
        with open(os.path.join(args.result_dir, "config.json"), "r") as f:
            config.update(json.load(f))
        args = argparse.Namespace(**config)
    else:
        args = convert_None_arguments_to_Nones(args)
        args.result_dir = get_result_direcotry_name(exp_name, args.debug, args.result_dir)
        if args.debug:
            args.n_epochs = 1
        if args.no_logger:
            args.logger = "EmptyLogger"
    return args


def convert_None_arguments_to_Nones(args: argparse.Namespace) -> argparse.Namespace:
    """Convert None arguments to default values."""
    for key, value in vars(args).items():
        if value == "None":
            setattr(args, key, None)
    return args


def float_or_none(value):
    if value.lower() == 'none':
        return None
    return float(value)