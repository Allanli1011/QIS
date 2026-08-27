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
- 策略 = 纯函数 `prices -> weights`，输出前 `normalize_gross`；组合级缩放走 `portfolio/construction.py`。
- 新标的加在 `config/universe.yaml`；期货标的尽量配 `carry_leg`（远月 RIC），
  供换月调整与 carry 策略使用。
- 改动 `config/*.yaml` 结构或模块职责时，同步更新 README 的架构说明。
- `data/*.parquet` 与 `reports/` 不入库。

## 数据源注意

- `lseg.data` 的 default session 是进程单例，复用 `qis.data.lseg.get_source()`。
- 不同 RIC 收盘列名不同，一律走 `LSEGSource.normalize()` 归一化。
- 连续合约未复权：换月窗口优先用 `v1` 序列背离法（LSEG 成交量拼接点），
  无 v1 回退成交量规则；窗口内收益取远月，见 `qis.data.roll`。
- 换月成本由引擎 `roll_mask` 计（|持仓|×2×单边费率）；v1 序列随 `qis fetch-data` 更新。
- API 出 JSON 前必须过 `qis.web.service._jsonable`：pandas pivot 等操作可能产出
  `pd.NA`（nullable dtype），直接序列化会变成 `{}` 污染前端。
