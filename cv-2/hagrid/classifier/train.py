import math
from collections import defaultdict
import logging
import os

from omegaconf import DictConfig, OmegaConf

import torch.nn as nn
import torch.utils
import torch.optim
import torch
from torch.utils.tensorboard import SummaryWriter

from hagrid.classifier.metrics import get_metrics
from hagrid.classifier.utils import collate_fn, add_metrics_to_tensorboard, add_params_to_tensorboard, save_checkpoint

from metaflow import S3

class TrainClassifier:
    """
    Gesture classification training pipeline:
        -initialize dataloaders
        for n epochs from training config:
            -run one epoch
            -eval on validation set
            - metrics calculation
            -save checkpoint
    """

    @staticmethod
    def eval(
        model: nn.Module,
        conf: DictConfig,
        epoch: int,
        test_loader: torch.utils.data.DataLoader,
        writer: SummaryWriter,
        mode: str = "valid",
        device: str = None
    ) -> float:
        """
        Evaluation model on validation set and metrics calc

        Parameters
        ----------
        model : nn.Module
            Model for eval
        conf : DictConfig
            Config with training params
        epoch : int
            Number of epoch
        test_loader : torch.utils.data.DataLoader
            Dataloader for sampling test data
        writer : SummaryWriter
            Tensorboard log writer
        mode : str
            Eval mode valid or test
        """
        f1_score = None
        if test_loader is not None:
            with torch.no_grad():
                model.eval()
                predicts, targets = defaultdict(list), defaultdict(list)
                for i, (images, labels) in enumerate(test_loader):
                    images = torch.stack(list(image.to(device if device is not None else conf.device) for image in images))
                    output = model(images)

                    for target in list(labels)[0].keys():
                        target_labels = [label[target] for label in labels]
                        predicts[target] += list(output[target].detach().cpu().numpy())
                        targets[target] += target_labels

                for target in targets.keys():
                    metrics = get_metrics(
                        torch.tensor(targets[target]), torch.tensor(predicts[target]), conf, epoch, mode,
                        writer=writer, target=target
                    )
                    if target == "gesture":
                        f1_score = metrics["f1_score"]
                    add_metrics_to_tensorboard(writer, metrics, epoch, "valid", target=target)
        return f1_score

    @staticmethod
    def run_epoch(
        model: nn.Module,
        epoch: int,
        device: str,
        optimizer: torch.optim.Optimizer,
        lr_scheduler_warmup: torch.optim.lr_scheduler.LinearLR,
        train_loader: torch.utils.data.DataLoader,
        writer: SummaryWriter
    ) -> None:
        """
        Run one training epoch with backprop

        Parameters
        ----------
        model : nn.Module
            Model for eval
        epoch : int
            Number of epoch
        device : str
            CUDA or CPU device
        optimizer : torch.optim.optimizer.Optimizer
            Optimizer
        lr_scheduler_warmup :
            Linear learning rate scheduler
        train_loader : torch.utils.data.DataLoader
            Dataloader for sampling train data
        writer : SummaryWriter
            Tensorboard log writer
        """
        criterion = nn.CrossEntropyLoss()
        model.train()

        lr_scheduler_params = lr_scheduler_warmup.state_dict()

        if writer is not None:
            optimizer_params = optimizer.param_groups[0]
            add_params_to_tensorboard(writer, optimizer_params, epoch, "optimizer", {"params"})
            not_logging = lr_scheduler_params.keys() - {"start_factor", "end_factor"}
            add_params_to_tensorboard(writer, lr_scheduler_params, epoch, "lr_scheduler", not_logging)

        for i, (images, labels) in enumerate(train_loader):

            step = i + len(train_loader) * epoch

            images = torch.stack(list(image.to(device) for image in images))
            output = model(images)
            loss = []
            accuracies = {target:[] for target in list(labels)[0].keys()}

            for target in list(labels)[0].keys():

                target_labels = [label[target] for label in labels]
                target_labels = torch.as_tensor(target_labels).to(device)
                predicted_labels = output[target]
                loss.append(criterion(predicted_labels, target_labels))
                accuracies[target] = torch.sum(predicted_labels.argmax(axis=1) == target_labels).item() / len(labels)

            loss = sum(loss)
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                logging.info("Loss is {}, stopping training".format(loss_value))
                exit(1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if lr_scheduler_warmup is not None:
                lr_scheduler_warmup.step()

            if writer is not None:
                writer.add_scalar(f"loss/train", loss_value, step)
                logging.info(f"Step {step}: Loss = {loss_value}")

    # @staticmethod
    def train(
        model: nn.Module,
        conf: DictConfig,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        checkpoint_model_every_epoch: bool = False,
        number_of_epochs: int = None,
        device: str = None,
        tensorboard_s3_prefix: str = '',
        always_upload_best_model : bool = False
    ) -> None:
        """
        Initialization and running training pipeline

        Parameters
        ----------
        model : nn.Module
            Model for eval
        conf : DictConfig
            Config with training params
        train_dataset : torch.utils.data.Dataset
            Custom train gesture classification dataset
        test_dataset : torch.utils.data.Dataset
            Custom test gesture classification dataset
        checkpoint_model_every_epoch : bool
            Should model be saved every epoch? 
        number_of_epochs : int 
            Number of epochs to train the model for
        device : str
            Device to move PyTorch data and model to
        tensorboard_s3_prefix : str
            Prefix of path to put tensorboard logs in, versioned by Metaflow run. 
        always_upload_best_model : bool
            Should the best model always be uploaded?
        """
        experiment_path = "experiments/{}".format(conf.experiment_name)
        log_dir = os.path.join(tensorboard_s3_prefix, experiment_path, "logs")
        writer = SummaryWriter(log_dir = log_dir)
        writer.add_text(f"model/name", conf.model.name)
        epochs = conf.train_params.epochs if number_of_epochs is None else number_of_epochs
        model = model.to(device if device is not None else conf.device)
        params = [p for p in model.parameters() if p.requires_grad]
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=conf.train_params.train_batch_size,
            num_workers=conf.train_params.num_workers,
            collate_fn=collate_fn,
            persistent_workers = True,
            prefetch_factor=conf.train_params.prefetch_factor,
            shuffle=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=conf.train_params.test_batch_size,
            num_workers=conf.train_params.num_workers,
            collate_fn=collate_fn,
            persistent_workers = True,
            prefetch_factor=conf.train_params.prefetch_factor,
        )
        optimizer = torch.optim.SGD(
            params,
            lr=conf.optimizer.lr,
            momentum=conf.optimizer.momentum,
            weight_decay=conf.optimizer.weight_decay
        )
        warmup_iters = min(1000, len(train_dataloader) - 1)
        lr_scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=conf.scheduler.start_factor, total_iters=warmup_iters
        )
        best_metric = -1.0
        conf_dictionary = OmegaConf.to_container(conf)
        for epoch in range(conf.model.start_epoch, epochs):
            logging.info(f"Epoch: {epoch}")
            TrainClassifier.run_epoch(
                model,
                epoch,
                device if device is not None else conf.device,
                optimizer,
                lr_scheduler_warmup,
                train_dataloader,
                writer
            )
            current_metric_value = TrainClassifier.eval(model, conf, epoch, test_dataloader, writer, device=device)
            if checkpoint_model_every_epoch:
                save_checkpoint(
                    output_dir = experiment_path, 
                    config_dict = conf_dictionary, 
                    model = model, 
                    optimizer = optimizer, 
                    epoch = epoch,
                    name = f"model_{epoch}"
                )
            if current_metric_value > best_metric:
                logging.info(f"Saving best model with metric: {current_metric_value}")
                save_checkpoint(
                    output_dir = experiment_path,
                    config_dict = conf_dictionary, 
                    model = model, 
                    optimizer = optimizer, 
                    epoch = epoch, 
                    name = "best_model"
                )
                best_metric = current_metric_value
                if always_upload_best_model:
                    path_to_best_model = os.path.join(experiment_path, 'best_model.pth')
                    best_model_location = os.path.join(tensorboard_s3_prefix, path_to_best_model)
                    if best_model_location.startswith('s3://'):
                        with S3(s3root = tensorboard_s3_prefix) as s3:
                            s3.put_files([(path_to_best_model, path_to_best_model)])
                            print("Best model checkpoint saved at {}".format(best_model_location))
        writer.flush()
        writer.close()
        print("""
        View Tensorboard results in your browser with this command:
            tensorboard --logdir={}
        """.format(log_dir))