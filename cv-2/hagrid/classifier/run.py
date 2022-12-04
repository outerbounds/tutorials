import logging
import argparse
from typing import Optional, Tuple
import os

from omegaconf import OmegaConf, DictConfig

from torch.utils.tensorboard import SummaryWriter
import torch.utils
import torch.optim

from hagrid.classifier.dataset import GestureDataset
from hagrid.classifier.preprocess import get_transform
from hagrid.classifier.train import TrainClassifier
from hagrid.classifier.utils import set_random_state, build_model, collate_fn

from metaflow import S3

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)


def _initialize_model(
    conf: DictConfig, 
    model_name : str = None, 
    checkpoint_path : str = None,
    device : str = None
):
    set_random_state(conf.random_state)
    num_classes = len(conf.dataset.targets)
    conf.num_classes = {"gesture": num_classes, "leading_hand": 2}
    # checkpoint_path = conf.model.get("checkpoint", None) if checkpoint_path is None else checkpoint_path
    if checkpoint_path is not None and checkpoint_path.startswith('s3://'):
        print("Downloading checkpoint from: {}".format(checkpoint_path))
        with S3() as s3:
            result = s3.get(checkpoint_path)
            checkpoint_path = "best_model.pth"
            os.rename(result.path, checkpoint_path)
    model = build_model(
        model_name = model_name if model_name is None else model_name,
        num_classes = num_classes,
        checkpoint = checkpoint_path,
        device = conf.device if device is None else device,
        pretrained = conf.model.pretrained,
        freezed = conf.model.freezed
    )
    return model


def run_test(
    path_to_config: str, 
    device:str = None, 
    model_name:str = None, 
    checkpoint_path:str = None,
    tensorboard_s3_prefix:str = ''
):
    """
    Run training pipeline

    Parameters
    ----------
    path_to_config : str
        Path to config
    """
    conf = OmegaConf.load(path_to_config)
    model = _initialize_model(conf, model_name, checkpoint_path, device)
    experiment_path = f"experiments/{model_name}"
    log_dir = os.path.join(tensorboard_s3_prefix, experiment_path, "logs")
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text(f'model/name', model_name)
    test_dataset = GestureDataset(is_train=False, conf=conf, transform=get_transform(), is_test=True)
    logging.info(f"Current device: {conf.device if device is None else device}")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=conf.train_params.test_batch_size,
        num_workers=conf.train_params.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=conf.train_params.prefetch_factor,
    )
    TrainClassifier.eval(model, conf, 0, test_dataloader, writer, "test")

def run_train(
    path_to_config: str, 
    number_of_epochs:int = None, 
    device:str = None, 
    model_name:str = None,
    checkpoint_path:str = None, 
    tensorboard_s3_prefix : str = '',
    always_upload_best_model : bool = False
) -> None:
    """
    Run training pipeline
    Parameters
    ----------
    path_to_config : str
        Path to config
    """
    conf = OmegaConf.load(path_to_config)
    if model_name != conf['experiment_name']:
        conf['experiment_name'] = model_name
    logging.info(f"Current device: {device if device is not None else conf.device}")
    model = _initialize_model(conf, model_name, checkpoint_path, device)
    train_dataset = GestureDataset(is_train=True, conf=conf, transform=get_transform())
    test_dataset = GestureDataset(is_train=False, conf=conf, transform=get_transform())
    TrainClassifier.train(
        model, conf, train_dataset, test_dataset, 
        number_of_epochs = number_of_epochs, device = device, 
        tensorboard_s3_prefix = tensorboard_s3_prefix,
        always_upload_best_model = always_upload_best_model)

    return model

def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture classifier...")
    parser.add_argument(
        "-c", "--command",
        required=True,
        type=str,
        help="Training or test pipeline",
        choices=('train', 'test')
    )
    parser.add_argument(
        "-p",
        "--path_to_config",
        required=True,
        type=str,
        help="Path to config"
    )
    known_args, _ = parser.parse_known_args(params)
    return known_args

if __name__ == '__main__':
    args = parse_arguments()
    if args.command == "train":
        run_train(args.path_to_config)
    elif args.command == "test":
        run_test(args.path_to_config)
