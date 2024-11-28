import argparse
import json
import multiprocessing as mp
import os
import os.path as osp
import random
from math import ceil
from statistics import mean
from time import time
from typing import Dict

import torch
from torch.functional import Tensor
from tqdm import tqdm


def random_select_scales(results):
    original_len = len(results)
    _results = {x["obj_code"]: [] for x in results}
    for res in tqdm(results, desc="Selecting scale", ncols=150, leave=False):
        obj_code = res["obj_code"]
        _results[obj_code].append(res)
    results = []
    for res in _results.values():
        num_res = len(res)
        idx = random.randint(0, num_res - 1)
        results.append(res[idx])
    print(f"Selected {len(results)} / {original_len} results to evaluate.")
    return results


def generate_cmds(results_path, gpus, args):
    with open(results_path) as rf:
        results = json.load(rf)
    total = len(results)
    num_procs = len(gpus)
    print(f"Evaluating {total} samples on {num_procs} gpus.")
    data_per_proc = ceil(total / num_procs)
    start_ends = [(i, i + data_per_proc) for i in range(0, total, data_per_proc)]
    command = f'python {osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "utils", "eval_utils.py")}'
    cmds = []
    for i, x in enumerate(start_ends):
        cmd = f"{command} -r {results_path} -s {x[0]} -e {x[1]} -g {gpus[i]}"
        cmds.append(cmd)
    for c in cmds:
        print(c)
    return cmds, start_ends


def run_cmd(cmd):
    if cmd[-1] == "\n":
        cmd = cmd[:-1]
    os.system(cmd)


def merge_results(results_path: str, start_ends):
    results_dir = osp.dirname(results_path)
    all_metrics = []
    all_metric_filenames = []
    for x in start_ends:
        filename = osp.join(results_dir, f"metrics_{x[0]}_{x[1]}.json")
        all_metric_filenames.append(filename)
        with open(filename) as rf:
            all_metrics.append(json.load(rf))
    final_metric = {"overall_max_pen": -1, "mean_obj": {}, "mean_overall": {}, "details": {}}
    overall_max_pens = []
    overall_metric = []
    mean_obj_metric = []
    for m in all_metrics:
        overall_max_pens.append(m["overall_max_pen"])
        details = m["object_details"]
        for obj_code_with_scale in details.keys():
            mean_obj_metric.append(details[obj_code_with_scale]["mean"])
            overall_metric.extend(details[obj_code_with_scale]["detail"])
        final_metric["details"].update(details)
    for k in ["q1", "pen", "valid_q1"]:
        final_metric["mean_obj"][k] = mean([x[k] for x in mean_obj_metric])
        final_metric["mean_overall"][k] = mean([x[k] for x in overall_metric])
    final_metric["overall_max_pen"] = max(overall_max_pens)

    for fn in all_metric_filenames:
        print(f"Removing {fn}")
        os.remove(fn)

    return final_metric


def pool_run(commands):
    _t1 = time()
    pool = mp.Pool(8)
    for cmd in commands:
        pool.apply_async(run_cmd, args=(cmd, ))
    pool.close()
    pool.join()
    _t2 = time()
    print(f"Finished in {_t2 - _t1:.2f}s")


class DiversityEvalator:
    joint_limits = [
        [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
        [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
        [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
        [0.0, 0.785], [-0.349, 0.349], [0.0, 1.571], [0.0, 1.571], [0.0, 1.571],
        [-1.047, 1.047], [0.0, 1.222], [-0.209, 0.209], [-0.524, 0.524], [-1.571, 0.0]
    ]  # (22, 2)

    def __init__(self, result_path: str, metric: Dict) -> None:
        with open(result_path) as rf:
            self.results = json.load(rf)
        self.metric = metric

    def calc_diversity(self):
        overall_preds = {"translations": [], "joint_angles": []}
        per_obj_diversity = {"var_translation": [], "var_joint_angle": [], "entropy_joints": []}
        for res in self.results:
            obj_code = res["obj_code"]
            scale = res["scale"]
            obj_code_with_scale = f"{obj_code}_{scale}"
            predictions = torch.tensor(res["predictions"], dtype=torch.float)

            translations = predictions[:, :3]
            joint_angles = predictions[:, -22:]

            mean_var_trans = torch.var(translations, dim=0, unbiased=True).mean().item()
            mean_var_joints = torch.var(joint_angles, dim=0, unbiased=True).mean().item()
            mean_entropy_joints = self._calc_entropy(joint_angles)

            # update per object metric detail
            obj_mean_metric: Dict = self.metric["details"][obj_code_with_scale]["mean"]
            obj_mean_metric.update({
                "var_translation": mean_var_trans,
                "var_joint_angle": mean_var_joints,
                "entropy_joints": mean_entropy_joints,
            })
            self.metric["details"][obj_code_with_scale]["mean"] = obj_mean_metric
            # record for mean-object metrics
            per_obj_diversity["var_translation"].append(mean_var_trans)
            per_obj_diversity["var_joint_angle"].append(mean_var_joints)
            per_obj_diversity["entropy_joints"].append(mean_entropy_joints)
            # record for overall metrics
            overall_preds["translations"].append(translations)
            overall_preds["joint_angles"].append(joint_angles)

        overall_diversity = {
            "mean_var_trans": torch.var(torch.cat(overall_preds["translations"]), dim=0, unbiased=True).mean().item(),
            "mean_var_joints": torch.var(torch.cat(overall_preds["joint_angles"]), dim=0, unbiased=True).mean().item(),
            "mean_entropy_joints": self._calc_entropy(torch.cat(overall_preds["joint_angles"])),
        }
        mean_object_diversity = {
            "mean_var_trans": mean(per_obj_diversity["var_translation"]),
            "mean_var_joints": mean(per_obj_diversity["var_joint_angle"]),
            "mean_entropy_joints": mean(per_obj_diversity["entropy_joints"]),
        }
        final_metric = {
            "diversity": {
                "overall": overall_diversity,
                "mean_object": mean_object_diversity,
            }
        }
        final_metric.update(self.metric)
        return final_metric

    def _calc_entropy(self, samples: Tensor) -> float:
        """
        1. divide the motion range of each joint into 100 bins
        2. calculate the distribution for each joint
        3. calculate the entropy for each joint
        4. average over joints

        Params:
            samples: A tensor of shape (N, 22)
        Returns:
            mean entropy over all joints
        """
        result = torch.empty(22, dtype=torch.float)
        for i in range(22):
            _samples = samples[:, i]
            prob = torch.histogram(_samples, 100, range=self.joint_limits[i])[0]
            prob /= prob.sum()
            distribution = torch.distributions.Categorical(probs=prob)
            result[i] = distribution.entropy()
        return result.mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_path", type=str)
    parser.add_argument("-g", "--gpus", nargs="+", type=int)
    parser.add_argument("-p", "--partial_scales", action="store_true", help="only randomly select {--num_scales} scales for each object")
    parser.add_argument("-n", "--num_scales", type=int, help="how many scales to select for each object")
    args = parser.parse_args()

    partial_scales = args.partial_scales and (args.num_scales is not None)
    results_dir = osp.dirname(args.result_path)
    if partial_scales:
        save_path = osp.join(results_dir, f"metrics_{args.num_scales}_scales.json")
    else:
        save_path = osp.join(results_dir, "metrics.json")
    if osp.isfile(save_path):
        print(f"{save_path} already exists")
        input("Press Enter to DELETE it and re-eval the results, or Ctrl-C to exit")
        os.remove(save_path)

    with open(args.result_path) as rf:
        results = json.load(rf)


    # results = results[::6]
    temp_result_path = osp.join(results_dir, "_temp_result.json")
    temp_results = results
    if partial_scales:
        print(f"Only test {args.num_scales} scales for each object")
        temp_results = random_select_scales(results)
    with open(temp_result_path, "w") as wf:
        json.dump(temp_results, wf)

    cmds, start_ends = generate_cmds(
        temp_result_path,
        args.gpus,
        args,
    )
    pool_run(cmds)
    performance_results = merge_results(temp_result_path, start_ends)

    # evaluate diversity
    # diversity = DiversityEvalator(temp_result_path, performance_results)
    # final_result = diversity.calc_diversity()

    # save final results
    with open(save_path, "w") as wf:
        json.dump(performance_results, wf, indent=4)

    print(f"Removing {temp_result_path}")
    os.remove(temp_result_path)
