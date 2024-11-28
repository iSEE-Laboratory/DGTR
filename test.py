import argparse
import json
import logging
import os
import os.path as osp
import random
import shutil
from ast import literal_eval
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.functional import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_datasets
from model import build_model
from utils.config_utils import EasyConfig


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Tester(object):
    def __init__(self, cfg: EasyConfig) -> None:
        # hyper-parameters
        if cfg.seed is not None:
            setup_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.ncols = getattr(cfg, "ncols", None)  # for tqdm
        cfg.model.aux_outputs = False

        # result saving
        self.epoch = torch.load(cfg.model.checkpoint_path, map_location="cpu")["epoch"]
        self.save_root = osp.join(cfg.save_root, "test_results", f"epoch_{self.epoch}")
        try:
            os.makedirs(self.save_root)
        except FileExistsError:
            print(f"{self.save_root} already exists. It seems that this checkpoint has been tested.")
            input("Press Enter to EMPTY it and recreate a NEW one, or Ctrl-C to exit")
            shutil.rmtree(self.save_root)
            os.makedirs(self.save_root)

        # logs
        self.log_path = osp.join(self.save_root, cfg.log_dir)
        os.makedirs(self.log_path, exist_ok=True)
        self._init_test_logger()
        self._record_cfg(cfg)

        # init functions
        self.model = self._build_model(cfg.model)
        self.test_loader = self._build_dataloaders(cfg.data)

    def _init_test_logger(self, log_name: str = "test") -> None:
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)

        # set two handlers
        fileHandler = logging.FileHandler(
            osp.join(self.log_path, f"{log_name}.log"), mode="w")
        fileHandler.setLevel(logging.DEBUG)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.DEBUG)

        # set formatter
        formatter = logging.Formatter(
            "[%(asctime)s] {%(filename)s} %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        consoleHandler.setFormatter(formatter)
        fileHandler.setFormatter(formatter)

        # add
        logger.addHandler(fileHandler)
        logger.addHandler(consoleHandler)

        logging.root = logger

    def _record_cfg(self, cfg: EasyConfig) -> None:
        # save cfg to yaml file
        cfg_dir = osp.join(self.save_root, "config")
        os.makedirs(cfg_dir, exist_ok=True)
        with open(osp.join(cfg_dir, "test_config.yaml"), "w") as wf:
            wf.write(yaml.dump(cfg.dict(), indent=2, allow_unicode=True))

    def _build_model(self, model_cfg: EasyConfig) -> nn.Module:
        logging.info("Loading model...")
        current_model = build_model(model_cfg)
        logging.info(current_model)
        # load checkpoint
        ckpt = model_cfg.checkpoint_path
        trained_state_dict = torch.load(ckpt, map_location=self.device)["model_state_dict"]
        model_state_dict = current_model.state_dict()
        # check state dict differences
        different = self._check_state_dict(trained_state_dict, model_state_dict)
        if different:
            input("State dict inconsistency detected. Press ENTER to continue or Ctrl-C to interrupt")
        else:
            logging.info("checkpoint state dict is CONSISTENT with model state dict")
        model_state_dict.update(trained_state_dict)
        current_model.load_state_dict(model_state_dict)
        logging.info(f"Loaded model weights from {ckpt}")
        return current_model.to(self.device)

    def _build_dataloaders(self, data_cfg: EasyConfig) -> Tuple[DataLoader, DataLoader]:
        logging.info("Preparing dataloaders...")
        _, _, test_set = build_datasets(data_cfg)
        assert data_cfg.test.batch_size == 1, "only support test batch size == 1"
        test_loader = DataLoader(
            test_set,
            batch_size=data_cfg.test.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            collate_fn=test_set.collate_fn,
            num_workers=data_cfg.test.num_workers,
        )
        logging.info(f"Num test samples: {len(test_set)}")
        return test_loader

    @staticmethod
    def _check_state_dict(trained: Dict, created: Dict):
        """
        check the consistency between checkpoint state dict and model state dict
        Params:
            trained: checkpoint state dict
            created: model state dict
        Return:
            different: a bool variable indicating if checkpoint keys and model keys are different
        """
        trained_keys = set(trained.keys())
        created_keys = set(created.keys())
        difference = {
            "checkpoint_only": trained_keys - created_keys,
            "created_model_only": created_keys - trained_keys,
        }
        different = False
        if difference["checkpoint_only"]:
            different = True
            logging.warning(f"These keys ONLY exist in checkpoint state dict:")
            for key in difference["checkpoint_only"]:
                logging.warning(key)
        if difference["checkpoint_only"]:
            different = True
            logging.warning(f"These keys ONLY exist in current model state dict:")
            for key in difference["create_model_only"]:
                logging.warning(key)
        return different

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

    @torch.no_grad()
    def test_one_iter(self, data: Dict):
        _, _, predictions = self.model(data)
        raw_predictions = predictions["outputs"]["hand_model_pose"]  # (num_queries, 28)
        matched_predictions = predictions["outputs"]["matched"]["hand_model_pose"]  # (K, 28)
        matched_targets = data["matched"]["hand_model_pose"]  # (K, 28)
        raw_results = {
            "obj_code": data["obj_code"][0],
            "scale": data["scale"][0],
            "predictions": raw_predictions.squeeze(0).cpu().numpy().tolist(),
        }
        matched_results = {
            "obj_code": data["obj_code"][0],
            "scale": data["scale"][0],
            "predictions": matched_predictions.cpu().numpy().tolist(),
            "targets": matched_targets.cpu().numpy().tolist(),
        }
        # assignments["obj_code"] = data["obj_code"][0]
        # assignments["scale"] = data["scale"][0]

        return raw_results, matched_results

    def test(self) -> float:
        self.model.eval()

        with tqdm(
            total=len(self.test_loader),
            desc=f"Testing Epoch {self.epoch}",
            ncols=self.ncols,
        ) as progress_bar:
            raw_results = []
            matched_results = []
            assignments = {}
            for data_dict in self.test_loader:
                data_dict = self._recursive_to_device(data_dict)
                _raw_result, _matched_result = self.test_one_iter(data_dict)
                raw_results.append(_raw_result)
                matched_results.append(_matched_result)
                # for k,v in _assignment.items():
                #     if torch.is_tensor(v):
                #         _assignment[k] = v.detach().cpu().numpy().tolist()
                # if _assignment["obj_code"] not in assignments:
                #     assignments[_assignment["obj_code"]] = {_assignment["scale"]: _assignment}
                # else:
                #     assignments[_assignment["obj_code"]][_assignment["scale"]] = _assignment
                progress_bar.update()
        # save test results
        raw_results_path = osp.join(self.save_root, "raw_results.json")
        with open(raw_results_path, "w") as wf:
            json.dump(raw_results, wf, indent=4)
        logging.info(f'Raw results have been saved to {raw_results_path}')

        matched_results_path = osp.join(self.save_root, "matched_results.json")
        with open(matched_results_path, "w") as wf:
            json.dump(matched_results, wf, indent=4)
        logging.info(f'Matched results have been saved to {matched_results_path}')

        # assignments_path = osp.join(self.save_root, "assignments.json")
        # with open(assignments_path, "w") as wf:
        #     json.dump(assignments, wf, indent=4)
        # logging.info(f'Assignments have been saved to {assignments_path}')

        # assignment_savepath = osp.join(self.save_root, "assignment"".json")
        # with open(assignment_savepath, "w") as wf:
        #     json.dump(self.model.criterion.assignments_save, wf, indent=4)
        # logging.info(f'Matched results have been saved to {assignment_savepath}')

        print("-" * 120)
        print("You can visualize them by running:")
        print(f"\"python ./vis/visualize_results.py -r {matched_results_path} -s -o 40\"")
        print("-" * 120)
        print("You can evaluate them by running:")
        print(f"\"python ./tools/evaluate.py -r {raw_results_path} --gpus 0 1 2 3 4 5 6 7 --partial_scales --num_scales 1\"")
        print("-" * 120)


def load_cfg():
    """
    config loading priority:
        train_cfg < test_cfg < other argparse arguments
    """
    args = parse_args()
    test_cfg = EasyConfig()
    # train cfg
    test_cfg.load(args.train_cfg)
    # test cfg
    new_cfg = EasyConfig()
    new_cfg.load(args.test_cfg)
    test_cfg.update(new_cfg.dict())

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
    assert len(args.override) % 2 == 0
    override_cfg = dict(zip(args.override[::2], args.override[1::2]))
    test_cfg.update(_load_argparse_cfg(override_cfg))
    return test_cfg


def parse_args(arg_str: Optional[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--train_cfg", type=str, required=True, help="config file used to train the model")
    parser.add_argument("-o", "--test_cfg", type=str, required=True, help="test config file, will overwrite the train config")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="override any config item (highest priority)",
        # example
        metavar="model.decoder.num_query 1",
    )
    args = parser.parse_args(arg_str)
    return args


if __name__ == "__main__":
    test_cfg = load_cfg()
    tester = Tester(test_cfg)
    tester.test()
