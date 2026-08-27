# QIS 策略研究回测平台

基于 LSEG（Refinitiv）行情数据的 QIS 日频策略研究与回测平台。
覆盖数据接入 → 本地缓存 → 信号/策略 → 组合构建 → 向量化回测 → 绩效与归因的完整链路。

## 环境前提

- **uv** 管理的 Python 3.13（`uv sync` 一键建环境）。
- **LSEG / Refinitiv Workspace 桌面端在线**（拉数依赖桌面会话）；
  可选环境变量 `LSEG_APP_KEY`（不设置则用库内置默认 key 连本机 Workspace）。

## 快速开始

```bash
uv sync                                # 建环境
uv run qis fetch-data                  # 从 LSEG 拉标的池日线，增量缓存到 data/*.parquet
uv run qis serve                       # 启动 Web 研究终端 http://127.0.0.1:8600
uv run qis run --strategy trend        # CLI 回测：trend / xsmom / carry
uv run qis run --strategy carry --attrib   # 附分标的盈亏归因
uv run qis report --returns reports/trend_returns.csv   # 重新生成 tearsheet
uv run pytest                          # 测试
```

**Web 终端**（`qis serve`）：总览看板 / 策略回测（参数可调：资产类别、波动目标、
起始日、成本开关）/ 标的池（搜索、筛选、单标的走势抽屉）/ 数据状态四个页面。

**标的池扩展**：`uv run python scripts/expand_universe.py` 会把候选 RIC 逐条向
LSEG 验证（拉 5 条日线），幸存者全量入库并重写 `config/universe.yaml`；
失败项记录在 `data/universe_validation.json`。

常用参数：`--start 2015-01-01`、`--vol-target 0.10`、`--no-cost`、`--tag exp1`。

## 架构

```
config/
  universe.yaml     标的池（98 个标的/8 类：股指、债券、外汇、能源、金属、农产品、利率、加密）
  settings.yaml     回测默认参数 + 按资产类别的成本 bps
src/qis/
  data/lseg.py      LSEG 适配器：session 复用、收盘字段归一化（TRDPRC_1/SETTLE/...）
  data/store.py     parquet 缓存，按最后日期增量更新
  data/roll.py      连续合约换月调整（成交量规则，见"数据说明"）
  data/universe.py  标的池加载
  strategy/         trend（时序动量）/ xsmom（截面动量）/ carry（期限结构）
  portfolio/        EWMA 波动、逆波动权重、组合波动目标
  backtest/engine.py 向量化日频引擎（防前视、换手与成本）
  analytics/        指标、tearsheet、归因
  web/              FastAPI 后端（service/api）+ 单页前端（static/，ECharts）
  cli.py            qis 命令行（fetch-data / run / report / serve）
scripts/
  expand_universe.py 候选 RIC 验证 + 全量拉取 + 重写 universe.yaml
  screenshot.py     Playwright 截图（视觉回归）
notebooks/          研究示例
tests/              pytest（引擎/指标/成本/缓存/换月/策略/Web API）
data/               parquet 缓存（gitignore）
reports/            回测输出（gitignore）
```

## 核心约定

- **防前视**：策略权重在 t 日收盘生成，引擎 `shift(1)` 后作用于 t→t+1 收益。
- **成本**：`cost = |Δw| × cost_bps`，Δw 考虑持仓随价格漂移；成本率按资产类别配置。
- **波动目标**：策略先输出毛敞口归一化（sum|w|=1）的权重，再按组合 EWMA 波动
  整体缩放到目标波动（默认 10%，杠杆上限 3×）。
- **无交易带**：目标权重偏离 < `rebal_band`（默认 3%）不调仓。实证日频微调既贡献
  换手、又对短期反转做反向交易（本池 trend：band 0→3% 使费前 Sharpe +0.17→+0.63）。

## 数据说明（重要）

- 连续合约（`c1/c2`）是**换月拼接、未复权**的价格：拼接日收益含新旧合约价差，
  不是持仓者真实损益。平台按数据可用性分两层识别换月窗口（`data/roll.py`）：
  1. **v1 背离法**（优先）：LSEG 的 `v1` 序列按成交量切换拼接，c1 与 v1 收益
     背离的窗口即官方换月点（CME 集团/JGB 等 19 个品种有 v1）；
  2. **成交量规则**（回退）：远月成交量超过近月的窗口。
  窗口内收益改用远月（c2）收益。
- **换月成本**：每个换月交易日按 |持仓| × 2 倍单边费率计"平旧开新"成本
  （引擎 `roll_mask` 参数），与调仓换手成本分开。
- FX 现货（`EUR=` 等）无换月问题，直接用收盘。
- 示例中三个策略是**研究模板**，不是可直接上实盘的产品策略。
  尤其朴素 calendar-spread carry 在强季节性品种（天然气等）上
  已被实证（含本平台归因）证明无效，需更专业的 carry 定义。

## 研究流程建议

1. `config/universe.yaml` 配标的 → `qis fetch-data`。
2. 在 `notebooks/` 里用 `qis` 包的模块迭代信号（见 01 号示例）。
3. 参数/结构定稿后落成 `strategy/` 里的函数，`qis run --strategy ... --tag ...` 归档结果。
