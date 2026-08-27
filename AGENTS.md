# AGENTS.md

## 项目

QIS 日频策略研究回测平台。uv 管理，Python 3.13，src layout（`src/qis/`）。

## 常用命令

- 测试：`uv run pytest tests/ -q`
- 拉数：`uv run qis fetch-data`（需本机 LSEG Workspace 在线）
- 回测：`uv run qis run --strategy {trend|xsmom|carry} [--attrib] [--no-cost]`
- 前端：`uv run qis serve`（FastAPI + `src/qis/web/static/` 单页应用，ECharts 已 vendor 在 static/vendor/）
- 扩品种：`uv run python scripts/expand_universe.py`（候选 RIC 逐条验证后才入库，禁止凭记忆写 RIC）
- 视觉回归：`uv run python scripts/screenshot.py`（需服务已启动 + dev 依赖 playwright）
- 环境：`uv sync`；加依赖用 `uv add` / `uv add --dev`

## 约定

- 语言与注释：代码注释/docstring 用中文（同现有风格）。
- 防前视是铁律：权重 t 日收盘生成、shift(1) 后生效；任何新逻辑只允许用 ≤t 的数据。
- **价格→收益一律走 `qis.data.panel.to_returns`**，不要直接 `prices.pct_change()`：
  价格矩阵是多市场交易日历的并集，标的假期是 NaN 洞，直接 pct_change 会把
  跨假期的涨跌整段丢掉（实测 98 个标的里 60 个累计收益偏差 >5pp）。
- 信号/波动目标/无交易带都在**完整历史**上算，`start` 只在最后切片时生效，
  否则回测窗口的第一年会被 lookback 预热吃掉。
- 策略 = 纯函数 `prices -> weights`，输出前 `normalize_gross`；组合级缩放走 `portfolio/construction.py`。
- xsmom 默认**全池排名 + 风险调整**，不要改回按资产类别分组：本池类别太小
  （rates/crypto 各 2 个会被整组置零），组内排名撑不起截面动量。
- carry 的 `c1/c2` 价差必须**两腿同日都有真实报价**才计（见 `carry_raw`），
  默认平滑 21 日 + 按合约到期间隔年化 + 截面模式——这个信号对日度测量噪声极其敏感。
- carry 优先用多月曲线的回归斜率（`curve_carry`），不是 c1/c2 两点差分：
  实测降噪 2~4 倍。补曲线数据用 `uv run python scripts/fetch_curve.py --depth 4`。
  能去掉近月就去掉（c1 到期收敛效应大），但金融期货的 c3/c4 更薄要留意。
- trend 默认 EWMAC（EWMA 快慢交叉），不要改回符号投票：后者丢掉趋势强度信息，
  实测 SR +0.70 vs +0.96 且换手高 74%。旧构造留在 `method="sign"`。
- 调参前先看 README「策略表现的三个已知成因」：收益低首先是 `gross` 没放大
  （波动目标够不着），不是信号问题。
- 新标的加在 `config/universe.yaml`；期货标的尽量配 `carry_leg`（远月 RIC），
  供换月调整与 carry 策略使用。
- 改动 `config/*.yaml` 结构或模块职责时，同步更新 README 的架构说明。
- `data/*.parquet` 与 `reports/` 不入库。

## 数据源注意

- `lseg.data` 的 default session 是进程单例，复用 `qis.data.lseg.get_source()`。
- 不同 RIC 收盘列名不同，一律走 `LSEGSource.normalize()` 归一化。
- 连续合约未复权：换月日用**持仓量（oi）跳升**识别，无 oi 回退成交量跳升；
  换月日收益取 `c1(T)/c2(T-1) − 1`（同一张合约），见 `qis.data.roll`。
  识别不出或频率离谱的标的由 `roll_diagnostics` 显式报出，不要静默吞掉。
- 换月成本由引擎 `roll_mask` 计（|持仓|×2×单边费率）。
- 原始行情有坏价（错价、单位错误、负结算），入回测前过 `qis.data.panel.clean_prices`。
- `scripts/expand_universe.py` 以 `config/universe.yaml` 已有条目为准，
  脚本里的 CANDIDATES 只用于新增；改标的 RIC 请直接改 yaml。
- API 出 JSON 前必须过 `qis.web.service._jsonable`：pandas pivot 等操作可能产出
  `pd.NA`（nullable dtype），直接序列化会变成 `{}` 污染前端。
