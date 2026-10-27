# 实验协议

## 原仓库数据

`data/raw/original/` 保存三个既有工作簿：

| 工作簿 | 工作表 | 原始列 | 数据点 |
|---|---|---|---:|
| simulated annealing.xlsx | Sheet1 | Time、AI、AV | 1000 |
| simulated annealing_bad.xlsx | Sheet1 | Time、AI、AV | 4096 |
| random number.xlsx | Sheet1 | Time、AI | 4096 |

前两个工作簿的 AV 分别为 −2 V、−2.3 V；第三个工作簿未记录电压。空白 Sheet2/Sheet3 不作为测量记录。

## R4C5

| 子目录 | 工作簿数 | 记录数 | 类型 |
|---|---:|---:|---|
| -0.25V IT | 1 | 1 | I–t |
| 0.01~2V IT | 1 | 10 | I–t |
| IT 1-1V CYCLE | 1 | 13 | I–t |
| fast -1v | 1 | 1 | I–t |
| IV BIPOLAR | 1 | 24 | I–V |

偏压使用数据列 AV。`160R4C5-0.25VA1V` 的 AV 实际为 +0.25 V，工作表原名保留。名称包含 `dead` 的记录仍列入全部统计，并通过 `name_contains_dead` 字段记录原始名称信息，不据名称判断器件状态。

Settings 中电压为 Programmed，Compliance=0.105 A。实际采样间隔由 Time 差分计算；普通 I–t 约 7.97 Hz，fast 记录约 9.27 Hz。逐记录设置见分析清单。

## R4C4

以下协议依据用户于 2026-10-26 提供及补充的信息。

| 工作簿目录 | Run 范围 | 顺序中的偏压 | 器件 | 测试时光照 |
|---|---|---|---|---|
| -1to1_test1 | 3164–3170 | −1、+1、−1、+1、−1、+1、−1 V | R4C4 | 无 |
| -1to1_test2 | 3172–3178 | −1、+1、−1、+1、−1、+1、−1 V | R4C4 | 无 |
| Light | 3179–3182 | −1、−1.5、+1、+1.5 V | 同基板另一器件 | 四段均有 |
| -1to1_test3 | 3183–3185 | −1、+1、−1 V | R4C4 | 无 |

- 总测试顺序：test1 → test2 → Light → test3。
- 每个 test 内为连续的交替偏压测试序列。
- 主器件 test1→test2：连续 0 V 钳位 14 min 4 s（844 s）。
- 主器件 test2→test3：连续 0 V 钳位 16 min（960 s）。中间 Light 测试在另一器件上进行，不从主器件的 960 s 中扣除。
- Light 中 ±1 V 为历史准备，随后对应 ±1.5 V 测试。用户记录的历史施压后松弛时间为 1 min 20 s（80 s）。
- Light 器件制造原理相同，内部可能不同；四段记录在分析中使用 `Light_other_device`，不与主器件拼接为一条状态轨迹。
- 用户记录光照可能影响 test3。主器件实际受光范围、光强和光谱未记录；test3 测试本身在暗处进行。
- test1 的 +1 V 波动可能与初始条件有关，这是用户提供的实验背景，未作为删除数据的条件。

各记录 4096 点，时长约 146.4 s。Settings 的 Interval=0.01 s；实际 Time 差分约 0.03575 s，对应约 27.97 Hz。Measure Voltage=Programmed，Current Range=Limited Auto=1uA，Compliance=0.105 A。

Run 编号和 Last Executed 用于确定记录先后。0 V 钳位时长采用用户记录；不由执行时标推断未记录区间的实际端口状态。时区未在原始仪器时间中提供，清单原样保留该时间字符串。
