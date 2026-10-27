import numpy as np
from scipy import signal, stats


def causal_residual(t, x, tau=2.):
    t, x = np.asarray(t, float), np.asarray(x, float)
    if tau <= 0 or len(t) != len(x) or np.any(np.diff(t) <= 0):
        raise ValueError('Positive tau and matching, strictly increasing times are required')
    baseline = x[0]
    residual = np.zeros_like(x)
    for k in range(1, len(x)):
        residual[k] = x[k] - baseline
        baseline += -np.expm1(-(t[k] - t[k-1]) / tau) * residual[k]
    return residual


def acf(x, maximum=200):
    x = np.asarray(x, float) - np.mean(x)
    denom = np.dot(x, x)
    if denom == 0:
        return np.r_[1., np.zeros(min(maximum, len(x)-1))]
    return np.array([np.dot(x[:len(x)-k], x[k:]) / denom for k in range(min(maximum, len(x)-1)+1)])


def summarize(r):
    x, v = r.current, r.voltage
    row = dict(dataset=r.dataset, device=r.device, group=r.group, source=r.source,
               sheet=r.sheet, kind='IT' if r.time is not None else 'IV', n=len(x),
               voltage_median_V=float(np.median(v)) if v is not None else None,
               voltage_min_V=float(v.min()) if v is not None else None,
               voltage_max_V=float(v.max()) if v is not None else None,
               current_mean_mA=float(x.mean()*1e3), current_std_uA=float(x.std()*1e6),
               current_min_mA=float(x.min()*1e3), current_max_mA=float(x.max()*1e3),
               acquired=r.settings.get('Last Executed'),
               voltage_kind=r.settings.get('Measure Voltage', 'unspecified'),
               name_contains_dead='dead' in r.sheet.lower())
    limit = r.settings.get('Compliance')
    row['compliance_A'] = float(limit) if limit else None
    row['at_99pct_compliance_fraction'] = float(np.mean(abs(x) >= .99*float(limit))) if limit else None
    if r.time is None:
        return row
    t = r.time - r.time[0]
    dt = np.median(np.diff(t))
    residual = causal_residual(t, x)
    kept = residual[t >= 10]
    if len(kept) < 3:
        raise ValueError(f'{r.key}: insufficient samples after 10 s')
    rho = acf(kept)
    stop = next((k for k in range(1, len(rho)) if rho[k] <= 0), len(rho))
    row.update(duration_s=float(t[-1]), dt_median_s=float(dt), sample_rate_Hz=float(1/dt),
               dt_relative_jitter=float(np.std(np.diff(t))/np.mean(np.diff(t))),
               residual_sigma_uA=float(kept.std()*1e6),
               residual_relative_pct=float(kept.std()/abs(x.mean())*100) if x.mean() != 0 else None,
               raw_lag1=float(acf(x, 1)[1]), residual_lag1=float(rho[1]),
               tau_effective_s=float(dt*(.5+rho[1:stop].sum())),
               acf_sum_truncated=stop == len(rho), residual_skew=float(stats.skew(kept)),
               residual_excess_kurtosis=float(stats.kurtosis(kept)),
               difference_mad_uA=float(stats.median_abs_deviation(np.diff(x), scale='normal')/np.sqrt(2)*1e6),
               max_abs_step_mA=float(np.max(abs(np.diff(x)))*1e3))
    return row


def filter_sensitivity(r):
    if r.time is None:
        return []
    t = r.time - r.time[0]
    output = []
    for tau in [.5, 2., 10.]:
        residual = causal_residual(t, r.current, tau)
        for label, start in [('five_tau', 5*tau), ('common_50s', 50.)]:
            x = residual[t >= start]
            if len(x) < 3:
                continue
            rho = acf(x)
            stop = next((k for k in range(1, len(rho)) if rho[k] <= 0), len(rho))
            output.append(dict(source=r.source, sheet=r.sheet, tau_s=tau, support=label,
                start_s=start, n=len(x), sigma_uA=float(x.std()*1e6),
                tau_effective_s=float(np.median(np.diff(t))*(.5+rho[1:stop].sum())),
                acf_sum_truncated=stop == len(rho)))
    return output


def windows(r):
    if r.time is None:
        return []
    t = r.time - r.time[0]
    x = r.current
    residual = causal_residual(t, x)
    spans = [('early', 10., 40.), ('late', 110., 140.)]
    spans += [('local10s', float(s), float(s+10)) for s in np.arange(10, t[-1]-10, 10)]
    out = []
    for name, start, end in spans:
        if t[-1] < end:
            continue
        ix = (t >= start) & (t < end)
        out.append(dict(source=r.source, sheet=r.sheet, window=name, start_s=start, end_s=end,
            n=int(ix.sum()), mean_mA=float(x[ix].mean()*1e3),
            residual_sigma_uA=float(residual[ix].std()*1e6),
            linear_sigma_uA=float(signal.detrend(x[ix]).std()*1e6)))
    return out


def iv_branches(r):
    if r.time is not None or r.voltage is None:
        return []
    out = []
    for v in [.5, -.5]:
        ix = np.flatnonzero(np.isclose(r.voltage, v, atol=1e-6))
        if len(ix) == 2:
            a, b = r.current[ix]
            den = (abs(a)+abs(b))/2
            out.append(dict(source=r.source, sheet=r.sheet, voltage_V=v,
                first_row=int(ix[0]+2), second_row=int(ix[1]+2), first_mA=float(a*1e3),
                second_mA=float(b*1e3), difference_mA=float(abs(a-b)*1e3),
                relative_difference=float(abs(a-b)/den) if den else None))
    return out
