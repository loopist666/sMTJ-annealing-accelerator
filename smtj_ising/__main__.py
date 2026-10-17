import argparse
from pathlib import Path
import sys

from .measurements import load_records
from .experiment import characterize_all, select_profiles, run_benchmark, image_demo, write_json, write_csv
from .reporting import figures, dashboard, chinese_report


def main():
    parser=argparse.ArgumentParser(description="R4C5 recorded-noise software Ising demo (no p-bit)")
    parser.add_argument("--data-dir",type=Path,required=True,help="R4C5 workbook folder; read-only")
    parser.add_argument("--out",type=Path,default=Path("results/r4c5_local"))
    parser.add_argument("--steps",type=int,default=2048)
    parser.add_argument("--runs",type=int,default=32,help="Software seeds per measured record and instance")
    parser.add_argument("--gain",type=float,default=1.5)
    parser.add_argument("--seed",type=int,default=700)
    parser.add_argument("--filter-tau",type=float,default=2.)
    parser.add_argument("--analyze-only",action="store_true")
    args=parser.parse_args()
    try:
        root=args.data_dir.expanduser().resolve()
        out=args.out.expanduser().resolve()
        if root == out or root in out.parents:
            raise ValueError("Output must be outside the read-only measurement directory")
        records=load_records(root)
        out.mkdir(parents=True,exist_ok=True)
        summary=characterize_all(records,out)
        print(f"Read {len(records)} records. Diagnostics: {out}",flush=True)
        if args.analyze_only:
            return
        profiles=select_profiles(records,args.filter_tau)
        print(f"Benchmark: {len(profiles)} measured -1 V records, {args.runs} seeds, {args.steps} attempts",flush=True)
        rows,result=run_benchmark(profiles,args.steps,args.runs,args.gain,args.seed)
        write_csv(out/"runs.csv",rows)
        write_csv(out/"benchmark_summary.csv",result["summary"])
        write_csv(out/"trace_summary.csv",result["by_trace"])
        write_json(out/"benchmark.json",result)
        write_json(out/"preprocessing.json",[{k:v for k,v in p.items() if k not in ("test_z","test_time","test_power_W")} for p in profiles])
        img=image_demo(profiles[0],args.steps,args.gain)
        write_json(out/"image_demo.json",img)
        figures(records,profiles,summary,result,img,out)
        dashboard(profiles,result,summary,out)
        chinese_report(summary,profiles,result,img,out)
        print(f"Done: {out/'demo.html'}\nReport: {out/'report_zh.md'}",flush=True)
    except (ValueError,FileNotFoundError) as exc:
        parser.exit(2,f"Error: {exc}\n")


if __name__ == "__main__":
    main()
