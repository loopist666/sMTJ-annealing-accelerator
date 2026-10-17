"""Fixed robustness grid, not tuning or a new primary performance claim."""
import argparse
from pathlib import Path
import numpy as np

from .measurements import load_records
from .experiment import select_profiles, run_benchmark, write_csv, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=Path("results/r4c5_local"))
    parser.add_argument("--runs",type=int,default=8)
    args=parser.parse_args()
    root,out=args.data_dir.resolve(),args.out.resolve()
    if root==out or root in out.parents:
        parser.error("Output must be outside the measurement directory")
    records=load_records(root)
    rows=[]
    for tau in (.5,2.,10.):
        profiles=select_profiles(records,tau)
        for gain in (.5,1.5,3.):
            _,result=run_benchmark(profiles,2048,args.runs,gain,700)
            for mode in ("greedy","constant","gaussian","replay","shuffled","sa"):
                group=[r for r in result["summary"] if r["mode"]==mode]
                rows.append(dict(filter_tau_s=tau,gain=gain,mode=mode,
                    software_runs=sum(r["n_runs"] for r in group),
                    mean_gap=float(np.mean([r["mean_gap"] for r in group])),
                    mean_hit_rate=float(np.mean([r["hit_rate"] for r in group]))))
            print(f"tau={tau}, gain={gain}: done",flush=True)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"solver_sensitivity.csv",rows)
    write_json(out/"solver_sensitivity.json",dict(runs_per_trace=args.runs,steps=2048,seed=700,
        scope="Fixed grid; four instances and eight traces reused. Not hyperparameter selection.",rows=rows))


if __name__=="__main__":main()
