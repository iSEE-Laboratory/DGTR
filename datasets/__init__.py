from typing import Optional, Tuple

from torch.utils.data import Dataset

from .dex_datatsets import DgnSetFull

def build_datasets(data_cfg) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    print(data_cfg.name.lower())
    if data_cfg.name.lower() == "dgnfull":
        if not hasattr(data_cfg, "test"):
            train_set = DgnSetFull(
                data_cfg.train.data_root,
                data_cfg.train.data_path,
                num_points=data_cfg.train.num_points,
                rotation_type=data_cfg.train.rotation_type,
                num_gt=data_cfg.train.num_gt,
                is_train = True,
                assignment_path=data_cfg.train.assignment
                
            )
            val_set = DgnSetFull(
                data_cfg.val.data_root,
                data_cfg.val.data_path,
                num_points=data_cfg.val.num_points,
                rotation_type=data_cfg.val.rotation_type,
                num_gt=data_cfg.val.num_gt,
                is_train = False,
            )
        # TODO compatible to new datatset
        else:
            train_set, val_set = None, None
        if hasattr(data_cfg, "test"):
            test_set = DgnSetFull(
                data_cfg.val.data_root,
                data_cfg.val.data_path,
                num_points=data_cfg.val.num_points,
                rotation_type=data_cfg.val.rotation_type,
                num_gt=data_cfg.val.num_gt,
                is_train = False
            )
        else:
            test_set = None
        return train_set, val_set, test_set
 
    else:
        raise NotImplementedError(f"Unable to build {data_cfg.name} dataset")
