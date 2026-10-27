from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
from urllib.parse import quote
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .io import load_records, sha256
from .metrics import summarize, causal_residual, acf, windows, filter_sensitivity, iv_branches


def dump_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def slug(record):
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', record.sheet)
    return name + '_' + hashlib.sha256(record.key.encode()).hexdigest()[:8]


def plot_record(r, destination):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    title = f'{r.dataset.upper()} / {r.device} / {r.sheet}'
    if r.dataset == 'original':
        title += '\n' + Path(r.source).name
    fig.suptitle(title, fontsize=13)
    x = r.current * 1e3
    if r.time is not None:
        t = r.time-r.time[0]
        z = causal_residual(t, r.current)*1e6
        mask = t >= 10
        axes[0,0].plot(t, x, lw=.6)
        axes[0,0].set(xlabel='Time from record start (s)', ylabel='Current (mA)', title='Recorded current')
        axes[0,1].plot(t[mask], z[mask], lw=.6, color='#ad6833')
        axes[0,1].set(xlabel='Time from record start (s)', ylabel='Residual (uA)', title='Causal EMA residual, tau=2 s')
        axes[1,0].hist(z[mask], bins=60, density=True, color='#487c9b')
        axes[1,0].set(xlabel='Residual (uA)', ylabel='Density (1/uA)', title='Residual distribution')
        rho = acf(z[mask])
        axes[1,1].plot(np.arange(len(rho))*np.median(np.diff(t)), rho)
        axes[1,1].axhline(0, color='grey', lw=.7)
        axes[1,1].set(xlabel='Lag (s)', ylabel='ACF', title='Residual autocorrelation')
    else:
        v = r.voltage
        axes[0,0].plot(v, x, '.-', ms=2, lw=.7)
        axes[0,0].set(xlabel='Voltage (V)', ylabel='Current (mA)', title='Recorded I-V path')
        axes[0,1].plot(np.arange(len(x)), v)
        axes[0,1].set(xlabel='Sample index', ylabel='Voltage (V)', title='Voltage sequence')
        axes[1,0].plot(np.arange(len(x)), x)
        axes[1,0].set(xlabel='Sample index', ylabel='Current (mA)', title='Current sequence')
        axes[1,1].hist(x, bins=40)
        axes[1,1].set(xlabel='Current (mA)', ylabel='Count', title='Samples across sweep')
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def spectrum(r):
    if r.time is None:
        return []
    t = r.time-r.time[0]
    z = causal_residual(t, r.current)[t >= 10]
    f, p = signal.welch(z, fs=1/np.median(np.diff(t)), nperseg=min(1024,len(z)), detrend='constant')
    return [dict(source=r.source, sheet=r.sheet, frequency_Hz=float(fq), residual_PSD_A2_per_Hz=float(pw)) for fq,pw in zip(f,p)]


def fmt(value):
    if value is None:
        return '未记录'
    return f'{value:.5g}' if isinstance(value, (int,float)) else str(value)


def generate(root, output, figures=True):
    records = load_records(root)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    all_stats = []
    manifest = []
    for path in sorted((root/'data/raw').rglob('*')):
        if path.is_file():
            manifest.append(dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha256(path)))
    dump_json(output/'file_manifest.json', manifest)
    sheet_manifest = [dict(source=r.source, sheet=r.sheet, dataset=r.dataset, device=r.device, group=r.group,
        sha256=r.sha256, n=len(r.current), data_rows_including_header=r.rows,
        columns=r.columns,
        settings=r.settings) for r in records]
    dump_json(output/'record_manifest.json', sheet_manifest)
    for dataset in ['original','r4c5','r4c4']:
        subset = [r for r in records if r.dataset == dataset]
        dest = output/dataset
        dest.mkdir(parents=True, exist_ok=True)
        stats_rows = [summarize(r) for r in subset]
        all_stats += stats_rows
        dump_csv(dest/'summary.csv', stats_rows)
        dump_csv(dest/'windows.csv', [w for r in subset for w in windows(r)])
        dump_csv(dest/'filter_sensitivity.csv', [w for r in subset for w in filter_sensitivity(r)])
        dump_csv(dest/'iv_branches.csv', [w for r in subset for w in iv_branches(r)])
        spec = [w for r in subset for w in spectrum(r)]
        dump_csv(dest/'spectra.csv', spec)
        report = [f'# {dataset.upper()} 测量统计', '',
            f'记录数：{len(subset)}。全部记录均列入；表中 σ 为 2 s 因果去基线后、距记录起点 10 s 以后的标准差。', '',
            '列定义和计算方法见 [方法说明](../../docs/methods.md)，实验条件见 [协议记录](../../docs/protocols.md)。', '',
            '| 组 / 器件 | 工作表 | 偏压中位数 V | 平均 I mA | 原始 σ μA | 残差 σ μA | τeff s | 时长 s |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for r,a in zip(subset,stats_rows):
            label = Path(r.source).name+' / '+r.sheet if dataset=='original' else r.sheet
            report.append('| '+' | '.join([r.group+' / '+r.device,label]+[fmt(a.get(k)) for k in ['voltage_median_V','current_mean_mA','current_std_uA','residual_sigma_uA','tau_effective_s','duration_s']])+' |')
        report += ['', '## 逐记录曲线', '']
        if figures:
            (dest/'figures').mkdir(exist_ok=True)
        for r in subset:
            name = slug(r)+'.png'
            if figures:
                plot_record(r, dest/'figures'/name)
            label = Path(r.source).name+' / '+r.sheet if dataset=='original' else r.sheet
            report += [f'### {label}', '', f'原始工作簿：[{Path(r.source).name}]({quote("../../"+r.source, safe="/")})；器件：{r.device}。', '', f'![{r.sheet}](figures/{name})', '']
        (dest/'README.md').write_text('\n'.join(report).rstrip()+'\n', encoding='utf-8')
        if figures and spec:
            fig, ax = plt.subplots(figsize=(10,6),layout='constrained')
            for r in subset:
                data = [w for w in spec if w['source']==r.source and w['sheet']==r.sheet]
                if data:
                    ax.loglog([w['frequency_Hz'] for w in data[1:]],[w['residual_PSD_A2_per_Hz'] for w in data[1:]],lw=.8,label=r.sheet)
            ax.set(xlabel='Frequency (Hz)',ylabel='Residual PSD (A^2/Hz)',title=f'{dataset.upper()}: 2 s EMA residual spectra')
            if len(subset) <= 8:
                ax.legend(fontsize=8)
            ax.grid(alpha=.15)
            fig.savefig(dest/'spectra.png',dpi=140);plt.close(fig)
    dump_csv(output/'all_measurements.csv', all_stats)
    main_rows = {a['sheet']:a for a in all_stats if a['device']=='R4C4'}
    pairs = []
    for first, second, wait in [('Run3170','Run3172',844),('Run3178','Run3183',960)]:
        a,b = main_rows[first],main_rows[second]
        pairs.append(dict(first_sheet=first,second_sheet=second,wait_0V_s=wait,
            first_mean_mA=a['current_mean_mA'],second_mean_mA=b['current_mean_mA'],
            first_residual_sigma_uA=a['residual_sigma_uA'],second_residual_sigma_uA=b['residual_sigma_uA'],
            sigma_ratio_second_over_first=b['residual_sigma_uA']/a['residual_sigma_uA'],
            prior_light_exposure='possible, dose unspecified' if wait==960 else 'not reported'))
    dump_csv(output/'r4c4/rest_pairs.csv', pairs)
    counts = {d:dict(workbooks=len({r.source for r in records if r.dataset==d}),
        records=sum(r.dataset==d for r in records), IT=sum(r.dataset==d and r.time is not None for r in records),
        IV=sum(r.dataset==d and r.time is None for r in records), samples=sum(len(r.current) for r in records if r.dataset==d)) for d in ['original','r4c5','r4c4']}
    dump_json(output/'inventory.json',counts)
    print(json.dumps(counts, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description='Generate descriptive measurement tables and figures')
    parser.add_argument('--root',type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--no-figures',action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root/'analysis'
    if output == root/'data' or (root/'data') in output.parents:
        parser.error('Analysis output must be outside data/')
    generate(root, output, not args.no_figures)


if __name__ == '__main__':
    main()
