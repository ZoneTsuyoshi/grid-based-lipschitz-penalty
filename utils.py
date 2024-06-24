import subprocess, time, json, requests, itertools, os, argparse, re
from typing import Union, List
import numpy as np
import torch
import logtools


def initialize(args: argparse.Namespace, exp_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    # go upstream while parent_dir exists
    if global_phase := args.parent_dir is None:
        args.n_model_steps = 1
    else:
        with open(os.path.join(args.parent_dir, "config.json"), "r") as f:
            config = json.load(f)
        args.n_model_steps = config["n_model_steps"] + 1
        args.downsampling = 1
        args.valid_ratio = config["valid_ratio"]
        args.test_ratio = config["test_ratio"]
        args.cross_validation = config["cross_validation"]
        args.split_seed = config["split_seed"]

    # set logger
    logger = getattr(logtools, args.logger)(f"{exp_name}-{args.n_model_steps}", args.experiment_key)

    # save config
    args.experiment_key = logger.get_key()
    if not (args.test or args.log):
        with open(os.path.join(args.result_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False)

    if not (args.test or args.log):
        logger.log_parameters(vars(args))

    return args, device, logger, global_phase


def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=("index", "uuid", "name", "timestamp", "memory.total", "memory.free", "memory.used", "utilization.gpu", "utilization.memory"), no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]


def is_japanese(string):
    # Regular expression that matches any of the Japanese unicode ranges:
    # Hiragana: U+3040 - U+309F
    # Katakana: U+30A0 - U+30FF
    # Kanji: U+4E00 - U+9FAF (Common + Uncommon Kanji)
    # Full-width Roman characters and Half-width Katakana: U+FF00 - U+FFEF
    # Katakana Phonetic Extensions: U+31F0 - U+31FF
    pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\uFF00-\uFFEF\u31F0-\u31FF]+'
    return re.search(pattern, string) is not None


def exclude_japanese(string, delimitter="-"):
    string = [tmp for tmp in string.split("/") if not is_japanese(tmp)]
    string = delimitter.join(string)
    return string


def post_slack_message(start_time: float, message: str, username:str, icon_emoji: str):
    elapsed_time = time.time() - start_time
    elapsed_hour = int(elapsed_time // (60*60))
    elapsed_min = int((elapsed_time - 60*60*elapsed_hour) // 60)
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    requests.post(slack_webhook_url,
              data = json.dumps({'text': message + "\n" + f"Elapsed time: {elapsed_hour}h{elapsed_min}m", 
                                 'username': username, 
                                 "icon_emoji": icon_emoji, 
                                 'link_names': 1,}))


def compute_convoluted_length(original_length: int, n_layers: int, kernel: Union[int, List[int]] = 3, pool_kernel_size: Union[int, List[int]] = 2, stride: Union[int, List[int]] = 1, padding: Union[int, List[int]] = 0, dilation: Union[int, List[int]] = 1) -> int:
    kernel, pool_kernel_size, stride, padding, dilation = [p if type(p)==list else list(itertools.repeat(p, n_layers)) for p in [kernel, pool_kernel_size, stride, padding, dilation]]
    current_length = original_length
    for i in range(n_layers):
        current_length = int((current_length + 2*padding[i] - dilation[i]*(kernel[i]-1)- 1) / stride[i] + 1) // pool_kernel_size[i]
    return current_length


def set_device(device: str):
    """Set device
    Args:
        device (str): Device to use
    
    Returns:
        device (torch.device): Device to use
    """
    if torch.cuda.is_available():
        if device=="cpu":
            device = torch.device("cpu")
        elif device=="gpu":
            device = torch.device("cuda")
        elif device.isdigit():
            device = torch.device("cuda", int(device))
        else:
            device = torch.device(device)
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    return device


def set_seed(seed: int):
    """Set seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_parent_arguments(args: argparse.Namespace):
    """Get parent arguments
    Args:
        args (argparse.Namespace): Arguments

    Returns:
        args (argparse.Namespace): parent arguments
        cargs (argparse.Namespace): child arguments
        local_phase (bool): Whether local phase or not
    """
    args.result_dir = args.result_dir.split("/", 1)[1]
    if local_phase := args.parent_dir is not None:
        cargs = args
        args.parent_dir = args.parent_dir.split("/", 1)[1]
        with open(os.path.join(args.parent_dir, "config.json"), "r") as f:
            config = json.load(f)
        args = argparse.Namespace(**config)
        args.result_dir = args.result_dir.split("/", 1)[1]
    else:
        cargs = None
    return args, cargs, local_phase