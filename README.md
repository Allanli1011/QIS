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
  data/panel.py     并集日历下的价格→收益（假期填充）、坏价清洗
  data/roll.py      连续合约换月调整（持仓量跳升法，见"数据说明"）
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
- **并集交易日历**：98 个标的来自不同市场，价格矩阵的索引是各自交易日历的并集，
  某标的自身的假期在矩阵里是 NaN 洞。收益一律走 `data/panel.py` 的 `to_returns`：
  存续区间内前值填充 → 假期当日收益 0、跨假期涨跌落到下一交易日；
  只有上市前/退市后算不可交易（持仓与目标权重同时清零，不产生换手）。
  **不要在别处直接对价格矩阵 `pct_change`**——那会把跨假期的涨跌整段丢掉。
- **信号在完整历史上算**：`start` 只在最后切片时生效。先切窗口会让 lookback
  预热吃掉回测的第一年（trend 需 252 日）。
- **成本**：`cost = |Δw| × cost_bps`，Δw 考虑持仓随价格漂移；成本率按资产类别配置。
- **波动目标**：策略先输出毛敞口归一化（sum|w|=1）的权重，再按组合 EWMA 波动
  整体缩放到目标波动（默认 10%，杠杆上限 `max_leverage` 默认 3×）。
  **注意本池的波动目标实际够不着**：98 个标的逆波动加权、gross=1 的未杠杆账本
  年化波动仅 ~1.3%（trend），打到 10% 需约 7.7×，EWMA(40) 在平静期估得更低、
  要 20×+。结果是上限在 90%+ 的交易日绑定，`--vol-target` 参数**不起作用**
  （调 15% 与 40% 结果完全相同）。回测输出与 `/api/run` 都会报告
  `leverage_cap_share`，绑定比例高就说明波动目标名存实亡。
  这是策略层面的风险预算问题——把上限直接调大并不能解决：组合收益被少数尾部日
  主导（VIX +112%、TTF +46%），实测上限放到 10× 会让 xsmom/carry 把净值打穿。
- **无交易带**：目标权重偏离 < `rebal_band`（默认 3%）不调仓，作用在加杠杆之后的
  实际持仓上。注意阈值是绝对权重单位，其相对强度随杠杆水平变化：把 `max_leverage`
  调高到上限不再绑定时，带会开始阻挡波动目标的降杠杆动作（实测 10× 下 trend
  波动被顶到 22%）。改杠杆要一并重标定这个阈值。

## 数据说明（重要）

- 连续合约（`c1/c2`）是**换月拼接、未复权**的价格：拼接日收益含新旧合约价差，
  不是持仓者真实损益。换月日靠**持仓量跳升**识别（`data/roll.py`）：
  c1 是拼接序列，换合约那天挂在它上面的 open interest 会从"即将到期、持仓已
  萎缩的旧合约"跳到新主力合约，量级往往翻数倍。实测这个信号双峰分离干净
  （ES/NQ 换月日 log 跳升 +0.8~1.5、平日 +0.01~0.06），抓到的日期落在到期月，
  频率也对得上（TY 4 次/年、CL 12 次/年）。无 oi 时回退成交量跳升。
- **换月日收益**用 `c1(T)/c2(T-1) − 1`：c1(T) 与 c2(T-1) 指的是同一张合约。
  （用 c2 自身收益是不对的——c2 当天同样滚动了，一样被污染。）
- **识别不了就说出来**：`roll_diagnostics` 给出方法与每年换月次数，
  次数离谱或识别不出的标的会在 CLI 输出和 Web 回测页显式列出，
  表示这些标的的收益**未被可靠的换月调整**，不再静默放过。
- **换月成本**：每个换月交易日按 |持仓| × 2 倍单边费率计"平旧开新"成本
  （引擎 `roll_mask` 参数），与调仓换手成本分开。
- **坏价清洗**（`data/panel.py`）：剔除非正价格（2020-04-20 WTI 结算 −13.1
  在比率收益口径下会给出 −172%）与"单日往返尖刺"（ROUGHRICE 2011-01-04 的
  1390.0 是 100 倍单位错误、HSCEI 2025-05-30 的 5380 是坏报价）。
  判据要求尖刺前后两天彼此自洽，因此 VIX 2018-02-05 的 +112% 这类真实行情不会被误伤。
- **数据稀疏**：部分序列本身缺数（NIYc1 只有 ~91 行/年，正常应 ~250），
  Web「数据状态」页有「行/年」列并标红，不要把这类标的的回测结果当真。
- FX 现货（`EUR=` 等）无换月问题，直接用收盘。
- 示例中三个策略是**研究模板**，不是可直接上实盘的产品策略。
  尤其朴素 calendar-spread carry 在强季节性品种（天然气等）上
  已被实证（含本平台归因）证明无效，需更专业的 carry 定义。

## 研究流程建议

1. `config/universe.yaml` 配标的 → `qis fetch-data`。
2. 在 `notebooks/` 里用 `qis` 包的模块迭代信号（见 01 号示例）。
3. 参数/结构定稿后落成 `strategy/` 里的函数，`qis run --strategy ... --tag ...` 归档结果。
