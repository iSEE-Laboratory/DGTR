import argparse
import logging
import os
import os.path as osp
import random
import shutil
import subprocess
import time
import json
from ast import literal_eval
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.functional import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import build_datasets
from model import build_model
from utils import EasyConfig

from test import Tester

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_git_head_hash():
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            git_hash = result.stdout.strip()
            return git_hash
        else:
            error_message = result.stderr.strip()
            return None
    except Exception as e:
        return None


class Trainer(object):
    def __init__(self, cfg: EasyConfig) -> None:
        # hyper-parameters
        assert cfg.device == cfg.model.device
        self.device = torch.device(cfg.device)
        self.ncols = getattr(cfg, "ncols", None)  # for tqdm
        self.num_epoch = cfg.epochs
        self.print_freq = cfg.print_freq
        self.resume = False
        if cfg.seed is not None:
            setup_seed(cfg.seed)
        if cfg.resume:
            assert osp.exists(cfg.checkpoint_path), f"FileNotFound: {cfg.checkpoint_path}"
            self.resume = True
            self.checkpoint_path = cfg.checkpoint_path
            self.checkpoint = torch.load(cfg.checkpoint_path, map_location=self.device)

        # model saving
        self.save_root = getattr(
            cfg,
            "save_root",
            f'Experiments/{cfg.save_comment}_{time.strftime("%Y%m%d%H%M%S", time.localtime())}'
        )
        if cfg.resume:
            self.save_root = osp.join(self.save_root, f'resume_from_{self.checkpoint["epoch"]}')
        self.saved_models = [
            {"save_path": "", "performance": -float('inf')}
        ] * cfg.save_top_n

        self.assignments = cfg.assignments

        # logs
        self.log_path = osp.join(self.save_root, cfg.log_dir)
        if not osp.exists(self.save_root):
            os.makedirs(self.save_root)
        if not osp.exists(self.log_path):
            os.makedirs(self.log_path)

        self.logger = SummaryWriter(log_dir=self.log_path)  # tensorboard logger
        self._init_train_logger()  # global Logging.logger
        self._record_cfg(cfg)
        # stat
        self.global_step = 0

        # init functions
        self.model = self._build_model(cfg.model)
        self.train_loader, self.val_loader = self._build_dataloaders(cfg.data)
        self.optimizer = self._build_optimizer(cfg.optimizer)
        self.scheduler = self._build_scheduler(cfg.scheduler)

    def _init_train_logger(self, log_name: str = "train") -> None:
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)

        # set two handlers
        fileHandler = logging.FileHandler(
            osp.join(self.log_path, f"{log_name}.log"), mode='w')
        fileHandler.setLevel(logging.DEBUG)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.DEBUG)

        # set formatter
        formatter = logging.Formatter(
            '[%(asctime)s] {%(filename)s} %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        consoleHandler.setFormatter(formatter)
        fileHandler.setFormatter(formatter)

        # add
        logger.addHandler(fileHandler)
        logger.addHandler(consoleHandler)

        logging.root = logger

    def _init_cfg_logger(self) -> None:
        logger = logging.getLogger("cfg")
        logger.setLevel(logging.DEBUG)

        # set two handlers
        fileHandler = logging.FileHandler(
            osp.join(self.log_path, "config.log"), mode='w')
        fileHandler.setLevel(logging.DEBUG)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.DEBUG)

        # set formatter
        formatter = logging.Formatter('%(message)s')
        consoleHandler.setFormatter(formatter)
        fileHandler.setFormatter(formatter)
        # add
        logger.addHandler(fileHandler)
        logger.addHandler(consoleHandler)

        return logger

    def _record_cfg(self, cfg: EasyConfig, display: bool = True) -> None:
        # save cfg to yaml file
        cfg_dir = osp.join(self.save_root, "config")
        os.makedirs(cfg_dir, exist_ok=True)
        with open(osp.join(cfg_dir, "train_config.yaml"), "w") as wf:
            wf.write(yaml.dump(cfg.dict(), indent=2, allow_unicode=True))

        git_hash = get_git_head_hash()
        if git_hash is not None:
            logging.info(f"Current git version: {git_hash}")
        else:
            logging.warning("No git repository detected")

        if display:
            def display_cfg(cfg: Dict, logger: logging.Logger):
                for k, v in cfg.items():
                    if isinstance(v, dict):
                        logger.info(f"\n{k}")
                        display_cfg(v, logger)
                    else:
                        logger.info(f"{k:<20s}: {v}")
            logger = self._init_cfg_logger()
            display_cfg(cfg, logger)
            logger.info('\n')

    def _build_optimizer(self, optim_cfg: EasyConfig) -> optim.Optimizer:
        optim_params = self.model.parameters()
        if optim_cfg.name.lower() == "adam":
            optimizer = optim.Adam(
                optim_params,
                lr=optim_cfg.lr,
                weight_decay=optim_cfg.weight_decay,
            )
        elif optim_cfg.name.lower() == "adamw":
            optimizer = optim.AdamW(
                optim_params,
                lr=optim_cfg.lr,
                weight_decay=optim_cfg.weight_decay,
            )
        else:
            raise NotImplementedError(f"No such optimizer: {optim_cfg.name}")
        if self.resume:
            optim_state_dict = self.checkpoint["optimizer_state_dict"]
            optimizer.load_state_dict(optim_state_dict)
        optimizer.zero_grad()
        return optimizer

    def _build_scheduler(self, sched_cfg: EasyConfig) -> optim.lr_scheduler._LRScheduler:
        self.last_epoch = self.checkpoint["epoch"] if self.resume else -1
        logging.info(f"Last epoch = {self.last_epoch}")
        # adjust learning rate
        if sched_cfg.name.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_cfg.t_max,
                eta_min=sched_cfg.min_lr,
                last_epoch=self.last_epoch,
            )
        elif sched_cfg.name.lower() == "steplr":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                sched_cfg.step_size,
                gamma=sched_cfg.step_gamma,
                last_epoch=self.last_epoch,
            )
        else:
            raise NotImplementedError(f"No such scheduler: {sched_cfg.name}")
        return scheduler

    def _build_model(self, model_cfg: EasyConfig) -> nn.Module:
        logging.info("Loading model...")
        current_model = build_model(model_cfg)
        logging.info(current_model)
        if self.resume:
            # load checkpoint
            trained_state_dict = self.checkpoint["model_state_dict"]
            model_state_dict = current_model.state_dict()
            # check state dict differences
            different = Tester._check_state_dict(trained_state_dict, model_state_dict)
            if different:
                input("State dict inconsistency detected. Press ENTER to continue or Ctrl-C to interrupt")
            else:
                logging.info("checkpoint state dict is CONSISTENT with model state dict")
            model_state_dict.update(trained_state_dict)
            current_model.load_state_dict(model_state_dict)
            logging.info(f"Loaded model weights from {self.checkpoint_path}")
        return current_model.to(self.device)

    def _build_dataloaders(self, data_cfg: EasyConfig) -> Tuple[DataLoader, DataLoader]:
        logging.info("Preparing dataloaders...")
        train_set, val_set, _ = build_datasets(data_cfg)
        train_loader = DataLoader(
            train_set,
            batch_size=data_cfg.train.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=False,
            collate_fn=train_set.collate_fn,
            num_workers=data_cfg.train.num_workers,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=data_cfg.val.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            collate_fn=val_set.collate_fn,
            num_workers=data_cfg.val.num_workers,
        )
        logging.info(f'Num train samples: {len(train_set)}')
        logging.info(f'Num val samples: {len(val_set)}')
        return train_loader, val_loader

    def _train_log(self, key: str, value: Union[float, int]) -> None:
        self.logger.add_scalar(
            f"train/{key}", value, global_step=self.global_step)

    def _val_log(self, key: str, value: Union[float, int], epoch: int = None) -> None:
        if epoch is not None:
            self.logger.add_scalar(f"val/{key}", value, global_step=epoch)
        else:
            self.logger.add_scalar(
                f"val/{key}", value, global_step=self.global_step)

    def _optimize_one_step(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _recursive_to_device(self, data: Union[Tensor, list, tuple, dict, Any]):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, list):
            for i in range(len(data)):
                data[i] = self._recursive_to_device(data[i])
            return data
        elif isinstance(data, tuple):
            new_data = []
            for i in range(len(data)):
                new_data[i] = self._recursive_to_device(data[i])
            return tuple(new_data)
        elif isinstance(data, dict):
            for k, v in data.items():
                data[k] = self._recursive_to_device(v)
            return data
        else:
            return data

    def _train_one_epoch(self, epoch: int) -> None:
        self.model.train()
        self.model.criterion.assignments_save = {}
        with tqdm(
            total=len(self.train_loader),
            desc=f"[Epoch {epoch}|{self.num_epoch}]",
            ncols=self.ncols
        ) as progress_bar:
            for batch_idx, data_dict in enumerate(self.train_loader, start=1):
                data_dict = self._recursive_to_device(data_dict)
                loss, loss_dict, _ = self.model(data_dict)

                loss.backward()
                self._optimize_one_step()

                # log
                self._train_log("loss", loss.item())
                for k, v in loss_dict.items():
                    self._train_log(k, v.item())

                if self.global_step % self.print_freq == 0:
                    logging.info(
                        f'{">" * 15} Epoch{epoch} [{batch_idx}|{len(self.train_loader)}] {"<" * 15}'
                    )
                    logging.info(f'{"Loss:":<21s} {loss.item():.4f} ')
                    for k, v in loss_dict.items():
                        logging.info(f'{k.title() + ":":<21s} {v.item():.4f}')

                # postfix string on progress bar
                postfix_str = f"loss={loss.item():.2f}"
                for k, v in loss_dict.items():
                    if k[-1].isdigit():
                        continue
                    postfix_str += f' {k.title():.5s}={v.item():.2f}'
                progress_bar.set_postfix_str(postfix_str, refresh=True)
                progress_bar.update()
                self.global_step += 1
        assignment_savepath = osp.join(self.save_root, "assignment_epoch_"+str(epoch)+".json")
        if self.assignments == "dynamic":
            with open(assignment_savepath, "w") as wf:
                json.dump(self.model.criterion.assignments_save, wf, indent=4)
            logging.info(f'Matched results have been saved to {assignment_savepath}')

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        self.model.eval()

        val_loss = []
        val_detailed_loss = {}

        with tqdm(
            total=len(self.val_loader),
            desc=f"Validating Epoch {epoch}",
            ncols=self.ncols,
            leave=False
        ) as progress_bar:
            for data_dict in self.val_loader:
                data_dict = self._recursive_to_device(data_dict)
                loss,  loss_dict, _ = self.model(data_dict)
                val_loss.append(loss.item())
                for k, v in loss_dict.items():
                    if k not in val_detailed_loss.keys():
                        val_detailed_loss[k] = [v.item()]
                    else:
                        val_detailed_loss[k].append(v.item())
                progress_bar.update()
        # log
        self._val_log("loss", mean(val_loss), epoch)
        for k, v in val_detailed_loss.items():
            self._val_log(k, mean(v), epoch)

        logging.info(f'{">" * 15} Epoch {epoch} Validated {"<" * 15}')
        logging.info(f'{"Loss:":<21s}: {mean(val_loss):.4f}')
        for k, v in val_detailed_loss.items():
            logging.info(f'{k.title() + ":":<21s}: {mean(v):.4f}')
        # return a float value representing self.model's current performance,
        # which is used for saving top n models.
        # Tips: LARGER value --> better performance
        return -mean(val_loss)

    def save_model(self, epoch: int, cur_performance: float, desc: str = "") -> None:
        save_path = osp.join(
            self.save_root, f"epoch{epoch}_minus_loss_{cur_performance:.4f}_{desc}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "minus_loss": cur_performance,
                "optimizer_state_dict": self.optimizer.state_dict()
            },
            save_path
        )
        return save_path

    def train(self) -> None:
        logging.info("Begin Training")

        performance = -999999
        latest_flag = "latest"
        begin_epoch = 1 if self.last_epoch == -1 else self.last_epoch + 1
        try:
            for ep in range(begin_epoch, self.num_epoch + 1):
                self._train_one_epoch(ep)
                performance = self._validate(ep)
                self.scheduler.step()

                # only save top n models
                # performances in self.saved_models is naturally and descendingly sorted
                for i, model in enumerate(self.saved_models):
                    if performance > model["performance"]:
                        save_path = self.save_model(ep, performance)
                        self.saved_models.insert(
                            i, {"save_path": save_path, "performance": performance})
                        del_path = self.saved_models.pop()["save_path"]
                        if del_path:
                            os.remove(del_path)
                        logging.info(
                            f"Top{i + 1} performance Reached: {performance:.4f} at epoch {ep}")
                        break
        except KeyboardInterrupt:
            print("Wait! Please allow me to save the latest model.\nJust a few seconds...")
            latest_flag = "interrupted"
        finally:
            self.save_model(ep, performance, latest_flag)
            logging.info(f"End of Training")

    def back_up(self) -> None:
        """
        back up a snapshot of current main codes
        """
        back_up_dir = osp.join(self.save_root, "backups")
        if not osp.exists(back_up_dir):
            os.makedirs(back_up_dir)
        back_up_list = [
            "datasets",
            "loss",
            "model",
            "tools",
            "utils",
            "vis",
            "multi_proc_test.py",
            "multi_proc_test.sh",
            "test.py",
            "test.sh",
            "train.py",
            "train.sh",
        ]
        back_up_list = [osp.join(osp.dirname(osp.abspath(__file__)), x) for x in back_up_list]
        back_up_list = list(filter(lambda x: osp.exists(x), back_up_list))
        for f in back_up_list:
            filename = osp.split(f)[-1]
            if osp.isdir(f):
                shutil.copytree(f, osp.join(back_up_dir, filename), symlinks=True, dirs_exist_ok=True)
            else:
                shutil.copy(f, back_up_dir, follow_symlinks=False)
            print(f"{filename} back-up finished")


def parse_args(arg_str: Optional[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--train_cfg", type=str, required=True)
    parser.add_argument("-o", "--other_cfg", type=str, nargs="*", default=[])
    parser.add_argument("-s", "--save_comment", type=str, default="",
                        help="save_comment will be showed in experiment folder")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="override any config item (highest priority)",
        # example
        metavar="model.decoder.num_query 1",
    )
    parser.add_argument("-r", "--resume", action="store_true")
    parser.add_argument("-c", "--checkpoint", type=str, help="checkpoint path")
    args = parser.parse_args(arg_str)
    return args


def load_cfg():
    """
    config loading priority:
        train_cfg < other_cfg < other argparse arguments
    """
    args = parse_args()
    train_cfg = EasyConfig()
    # train cfg
    train_cfg.load(args.train_cfg)
    # other cfg
    for c in args.other_cfg:
        new_cfg = EasyConfig()
        new_cfg.load(c)
        train_cfg.update(new_cfg.dict())

    def _load_argparse_cfg(args_dict: Dict[str, Any]):
        output = {}
        for key, arg in args_dict.items():
            arg = literal_eval(arg)
            if "." in key:
                split_keys = key.split(".")
                set_key = split_keys.pop()
                current_dict = output
                for k in split_keys:
                    if k not in current_dict.keys():
                        current_dict[k] = {}
                    current_dict = current_dict[k]
                current_dict[set_key] = arg
            else:
                output[key] = arg
        return output

    # argparse cfg
    train_cfg.update({
        "config_paths": [args.train_cfg] + args.other_cfg,
        "resume": args.resume,
        "checkpoint_path": args.checkpoint,
    })
    assert len(args.override) % 2 == 0

    override_cfg = dict(zip(args.override[::2], args.override[1::2]))

    train_cfg.update(_load_argparse_cfg(override_cfg))
    train_cfg.update({"save_comment": args.save_comment})
    return train_cfg


if __name__ == "__main__":
    train_cfg = load_cfg()
    pt_trainer = Trainer(train_cfg)
    pt_trainer.back_up()
    pt_trainer.train()
