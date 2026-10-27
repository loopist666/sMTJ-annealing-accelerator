# 电学测量数据与分析

本仓库保存原仓库测量数据、R4C5 和 R4C4 测试记录，以及对应的描述性统计和分析代码。内容包括原始工作簿、仪器导出的曲线图片、实验协议、时序、分布、自相关、频谱和 I–V 分支统计。

## 数据目录

| 数据集 | 工作簿 | 测量记录 | I–t | I–V | 原始数据 | 分析结果 |
|---|---:|---:|---:|---:|---|---|
| 原仓库数据 | 3 | 3 | 3 | 0 | [original](data/raw/original/) | [统计与曲线](analysis/original/README.md) |
| R4C5 | 5 | 49 | 25 | 24 | [r4c5](data/raw/r4c5/) | [统计与曲线](analysis/r4c5/README.md) |
| R4C4 | 4 | 21 | 21 | 0 | [r4c4](data/raw/r4c4/) | [统计与曲线](analysis/r4c4/README.md) |

R4C4 的 test1、test2、test3 为同一器件，共 17 条记录。`Light` 为同基板另一器件，共 4 条记录。两者在表格中使用不同的 `device` 标识。

## 阅读入口

- [实验协议与记录对应关系](docs/protocols.md)
- [分析方法、单位与字段定义](docs/methods.md)
- [按实验条件整理的数值记录](docs/observations.md)
- [所有测量的汇总表](analysis/all_measurements.csv)
- [原始文件 SHA256 清单](analysis/file_manifest.json)
- [工作表、仪器设置与器件标识](analysis/record_manifest.json)

原始工作簿及随附 PNG 按源文件字节保存。文件名保持原始命名，名称中的文字不作为分析分类或评价标签。临时锁文件及已解压数据的重复 ZIP 不重复收录。源数据不会在分析过程中改写。

## 复算

Python 3.11 或更新版本：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m measurement_analysis
python -m unittest discover -s tests -v
```

默认输出到 `analysis/`，包含各数据集的 CSV、每条记录的四联图和频谱图。另存到其他目录：

```bash
python -m measurement_analysis --output /tmp/measurement-analysis
```

图中文字采用英文及源记录名称，文档采用中文。CSV 中缺失的电压或时间字段留空；I–V 记录没有时间列时不计算时序统计。

## 仓库结构

```text
data/raw/               原始 Excel 与仪器图片
docs/                   协议、方法、数值记录
measurement_analysis/   读取与描述性分析代码
analysis/               可复算的统计表、图和来源清单
tests/                  源文件、读取和数值回归检查
```
