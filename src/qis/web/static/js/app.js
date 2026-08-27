/* ============ QIS Terminal 前端 ============ */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

/* Bloomberg 风配色 */
const C = {
  amber: "#f7a600", amberBright: "#ffc233",
  green: "#00c853", red: "#ff433d",
  cyan: "#29b6f6", violet: "#ab47bc", blue: "#4f8ef7",
  teal: "#26a69a", pink: "#f06292", orange: "#ff7043",
  dim: "#9a9a9a", faint: "#5c5c5c", border: "#262626",
};
const CLASS_COLORS = {
  "股指": C.blue, "债券": C.violet, "外汇": C.cyan, "能源": C.orange,
  "金属": C.amber, "农产品": C.green, "利率": C.teal, "加密": C.pink, "其他": C.dim,
};
const STRATEGY_NAMES = { trend: "时序趋势", xsmom: "截面动量", carry: "期限结构 Carry" };
const CLASS_ORDER = ["equity_index", "bond", "fx", "energy", "metal", "ags", "rates", "crypto"];
const CLASS_LABELS = {
  equity_index: "股指", bond: "债券", fx: "外汇", energy: "能源",
  metal: "金属", ags: "农产品", rates: "利率", crypto: "加密",
};

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

/* ---------- 格式化 ---------- */
const fmtPct = (v, d = 1) => v == null ? "—" : (v * 100).toFixed(d) + "%";
const fmtPct2 = (v) => v == null ? "—" : (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%";
const fmtNum = (v, d = 2) => v == null ? "—" : Number(v).toFixed(d);
const fmtSign = (v, d = 2) => v == null ? "—" : (v >= 0 ? "+" : "") + Number(v).toFixed(d);
const posNeg = (v) => v == null ? "" : v > 0 ? "pos" : v < 0 ? "neg" : "";
/* 行情价：精度按价格量级定，null 安全 */
const fmtPx = (v) => v == null ? "—" : Math.abs(v) < 20 ? fmtNum(v, 4) : fmtNum(v, 2);

/* ---------- SVG 迷你走势 ---------- */
function sparkline(points, { w = 130, h = 30, color = C.amber } = {}) {
  const ys = points.filter(v => v != null);
  if (ys.length < 2) return "";
  const min = Math.min(...ys), max = Math.max(...ys);
  const span = max - min || 1;
  const step = w / (ys.length - 1);
  const coords = ys.map((v, i) => `${(i * step).toFixed(1)},${(h - 3 - (v - min) / span * (h - 6)).toFixed(1)}`);
  const line = coords.join(" ");
  const lastUp = ys[ys.length - 1] >= ys[0];
  const c = color === "auto" ? (lastUp ? C.green : C.red) : color;
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
    <polygon points="0,${h} ${line} ${w},${h}" fill="${c}14"/>
    <polyline points="${line}" fill="none" stroke="${c}" stroke-width="1.3" stroke-linejoin="round"/>
  </svg>`;
}

/* ---------- ECharts 公共 ---------- */
const charts = new Map();
function chart(id) {
  const el = document.getElementById(id);
  if (!el) return null;
  if (charts.has(id)) return charts.get(id);
  const c = echarts.init(el, null, { renderer: "canvas" });
  charts.set(id, c);
  return c;
}
window.addEventListener("resize", () => charts.forEach(c => c.resize()));

const MONO = "ui-monospace, Menlo, monospace";
const baseAxis = {
  axisLine: { lineStyle: { color: C.border } },
  axisLabel: { color: C.dim, fontFamily: MONO, fontSize: 10 },
  splitLine: { lineStyle: { color: "#141414" } },
};
const baseTooltip = {
  trigger: "axis",
  backgroundColor: "#111111", borderColor: C.border,
  textStyle: { color: "#e8e8e8", fontSize: 11.5, fontFamily: MONO },
  axisPointer: { lineStyle: { color: C.faint } },
};

function lineOption(dates, series, { log = false, pct = false, area = true } = {}) {
  return {
    grid: { left: 56, right: 14, top: 14, bottom: 26 },
    tooltip: { ...baseTooltip, valueFormatter: v => pct ? fmtPct(v) : fmtNum(v, 4) },
    xAxis: { type: "time", ...baseAxis },
    yAxis: { type: log ? "log" : "value", scale: true, ...baseAxis,
             axisLabel: { ...baseAxis.axisLabel, formatter: v => pct ? (v * 100).toFixed(0) + "%" : v } },
    series: series.map(s => ({
      type: "line", showSymbol: false, smooth: false, ...s,
      data: s.points,
      lineStyle: { width: 1.4, color: s.color },
      itemStyle: { color: s.color },
      areaStyle: area ? { color: s.color + "12" } : undefined,
    })),
  };
}

/* ---------- 顶栏时钟 ---------- */
function startClock() {
  const tick = () => {
    const d = new Date();
    const p = n => String(n).padStart(2, "0");
    $("#tb-clock").textContent = `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
    $("#tb-date").textContent = `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
  };
  tick();
  setInterval(tick, 1000);
}

/* ---------- 行情滚动带 ---------- */
async function initTicker() {
  try {
    const inst = await api("/api/instruments");
    const items = inst.filter(i => i.chg_1d != null).map(i => {
      const cls = i.chg_1d >= 0 ? "up" : "dn";
      const px = fmtPx(i.last);
      return `<span class="tick-item"><b>${i.name}</b>${px} <span class="${cls}">${fmtPct2(i.chg_1d)}</span></span>`;
    });
    const html = items.join(`<span class="tick-sep">▪</span>`);
    $("#ticker").innerHTML = html + `<span class="tick-sep">▪</span>` + html;  // 两份实现无缝滚动
  } catch (_) { /* 后端未就绪时静默 */ }
}

/* ---------- 路由 ---------- */
$$(".nav-item").forEach(btn => btn.addEventListener("click", () => {
  $$(".nav-item").forEach(b => b.classList.toggle("active", b === btn));
  $$(".page").forEach(p => p.classList.toggle("active", p.id === "page-" + btn.dataset.page));
  if (btn.dataset.page === "universe") initUniverse();
  if (btn.dataset.page === "data") initData();
  if (btn.dataset.page === "strategy") ensureStrategyRun();
  setTimeout(() => charts.forEach(c => c.resize()), 50);
}));

/* ---------- 总览 ---------- */
const state = { overview: null, instruments: null, runs: {}, lastRunKey: null };

async function initOverview() {
  try {
    state.overview = await api("/api/overview");
    $("#conn-dot").classList.add("ok");
    $("#conn-text").textContent = "LIVE";
  } catch (e) {
    $("#conn-dot").classList.add("err");
    $("#conn-text").textContent = "OFFLINE";
    return;
  }
  const ov = state.overview;
  $("#overview-sub").textContent =
    `${ov.n_instruments} 标的 · ${ov.n_rics} RIC · 回测自 ${ov.settings.start} · 波动目标 ${(ov.settings.vol_target * 100).toFixed(0)}%`;

  const stats = [
    ["标的数", ov.n_instruments, "覆盖 " + ov.asset_classes.length + " 个资产类别"],
    ["RIC 数", ov.n_rics, "含 carry 远月腿"],
    ["资产类别", ov.asset_classes.length, ov.asset_classes.map(c => CLASS_LABELS[c] || c).join(" / ")],
    ["策略模板", ov.strategies.length, ov.strategies.map(s => STRATEGY_NAMES[s]).join(" / ")],
  ];
  $("#stat-row").innerHTML = stats.map(([l, v, s]) =>
    `<div class="kpi"><div class="kpi-label">${l}</div>
     <div class="kpi-value">${v}</div><div class="kpi-sub">${s}</div></div>`).join("");

  // 策略卡片（并行加载）
  const cardsEl = $("#strategy-cards");
  cardsEl.innerHTML = ov.strategies.map(s =>
    `<div class="skeleton" style="height:180px" id="card-${s}"></div>`).join("");
  const results = await Promise.allSettled(ov.strategies.map(s => api(`/api/run?strategy=${s}`)));
  results.forEach((res, i) => {
    const s = ov.strategies[i];
    const el = document.getElementById("card-" + s);
    if (res.status !== "fulfilled") {
      el.outerHTML = `<div class="strat-card"><div class="strat-head"><span class="strat-name">${STRATEGY_NAMES[s]}</span></div>
        <div class="strat-body" style="color:${C.red}">${res.reason.message}</div></div>`;
      return;
    }
    const d = res.value;
    state.runs[s] = d;
    const m = d.metrics;
    const eq = d.equity.map(p => p[1]);
    const shrink = eq.filter((_, j) => j % Math.ceil(eq.length / 240) === 0);
    el.outerHTML = `
      <div class="strat-card" onclick="gotoStrategy('${s}')">
        <div class="strat-head">
          <span class="strat-name">${STRATEGY_NAMES[s]}</span>
          <span class="strat-sharpe ${m.sharpe < 0 ? "neg" : ""}">SR ${fmtSign(m.sharpe)}</span>
        </div>
        <div class="strat-body">
          <div class="strat-spark">${sparkline(shrink, { w: 360, h: 72, color: m.ann_return >= 0 ? C.green : C.red })}</div>
          <div class="strat-metrics">
            <div class="sm"><span class="sm-label">年化收益</span><span class="sm-value ${posNeg(m.ann_return)}">${fmtPct(m.ann_return)}</span></div>
            <div class="sm"><span class="sm-label">年化波动</span><span class="sm-value">${fmtPct(m.ann_vol)}</span></div>
            <div class="sm"><span class="sm-label">最大回撤</span><span class="sm-value neg">${fmtPct(m.max_drawdown)}</span></div>
            <div class="sm"><span class="sm-label">Calmar</span><span class="sm-value">${fmtNum(m.calmar)}</span></div>
          </div>
        </div>
      </div>`;
  });

  // 资产分布
  const inst = await api("/api/instruments");
  state.instruments = inst;
  const byClass = {};
  inst.forEach(i => { const k = i.class_label; byClass[k] = (byClass[k] || 0) + 1; });
  const entries = Object.entries(byClass).sort((a, b) => b[1] - a[1]);
  chart("ov-class-chart").setOption({
    grid: { left: 40, right: 20, top: 16, bottom: 26 },
    tooltip: baseTooltip,
    xAxis: { type: "category", data: entries.map(e => e[0]), ...baseAxis },
    yAxis: { type: "value", ...baseAxis },
    series: [{
      type: "bar", data: entries.map(e => ({ value: e[1], itemStyle: { color: (CLASS_COLORS[e[0]] || C.dim) + "d9" } })),
      barWidth: "46%",
      label: { show: true, position: "top", color: C.dim, fontFamily: MONO, fontSize: 10 },
    }],
  }, true);
}

function gotoStrategy(s) {
  $$(".nav-item").find(b => b.dataset.page === "strategy").click();
  $$("#ctl-strategy .seg-btn").forEach(b => b.classList.toggle("active", b.dataset.value === s));
  ensureStrategyRun(true);
}

/* ---------- 策略回测页 ---------- */
let classesReady = false;
function buildClassChips() {
  if (classesReady || !state.overview) return;
  classesReady = true;
  const cls = [...state.overview.asset_classes].sort(
    (a, b) => CLASS_ORDER.indexOf(a) - CLASS_ORDER.indexOf(b));
  $("#ctl-classes").innerHTML = cls.map(c =>
    `<button class="chip" data-value="${c}">${CLASS_LABELS[c] || c}</button>`).join("");
  $$("#ctl-classes .chip").forEach(ch => ch.addEventListener("click", () => ch.classList.toggle("active")));
}

function selectedClasses() {
  return $$("#ctl-classes .chip.active").map(c => c.dataset.value);
}
function currentStrategy() {
  return $("#ctl-strategy .seg-btn.active").dataset.value;
}

$$("#ctl-strategy .seg-btn").forEach(b => b.addEventListener("click", () => {
  $$("#ctl-strategy .seg-btn").forEach(x => x.classList.toggle("active", x === b));
  ensureStrategyRun(true);
}));
$("#ctl-vt").addEventListener("input", () => $("#ctl-vt-label").textContent = $("#ctl-vt").value + "%");
$("#ctl-band").addEventListener("input", () => $("#ctl-band-label").textContent = $("#ctl-band").value + "%");
$("#ctl-run").addEventListener("click", () => ensureStrategyRun(true));

function runParams() {
  const cls = selectedClasses();
  return {
    strategy: currentStrategy(),
    start: $("#ctl-start").value || null,
    vol_target: $("#ctl-vt").value / 100,
    band: $("#ctl-band").value / 100,
    with_cost: $("#ctl-cost").checked,
    classes: cls.length ? cls.join(",") : null,
  };
}

async function ensureStrategyRun(force = false) {
  buildClassChips();
  const p = runParams();
  const qs = new URLSearchParams({ strategy: p.strategy });
  if (p.start) qs.set("start", p.start);
  if (p.vol_target) qs.set("vol_target", p.vol_target);
  qs.set("band", p.band);
  if (!p.with_cost) qs.set("with_cost", "false");
  if (p.classes) qs.set("classes", p.classes);
  const key = qs.toString();
  if (!force && state.lastRunKey === key) return;
  state.lastRunKey = key;

  $("#run-status").textContent = "计算中…";
  $("#status-msg").textContent = "RUNNING";
  $("#ctl-run").disabled = true;
  try {
    const d = await api("/api/run?" + key);
    renderRun(d);
    $("#run-status").textContent = "";
    $("#status-msg").textContent = "READY";
  } catch (e) {
    $("#run-status").textContent = "错误：" + e.message;
    $("#status-msg").textContent = "ERROR";
    setTimeout(() => { $("#run-status").textContent = ""; }, 4000);
  } finally {
    $("#ctl-run").disabled = false;
  }
}

function renderRun(d) {
  const m = d.metrics;
  $("#equity-title").textContent =
    `净值曲线 · ${STRATEGY_NAMES[d.strategy]} · 费后 vs 费前`;
  const defs = [
    ["年化收益", fmtPct(m.ann_return), posNeg(m.ann_return)],
    ["年化波动", fmtPct(m.ann_vol), ""],
    ["Sharpe", fmtSign(m.sharpe), posNeg(m.sharpe)],
    ["Sortino", fmtSign(m.sortino), posNeg(m.sortino)],
    ["最大回撤", fmtPct(m.max_drawdown), "neg"],
    ["Calmar", fmtNum(m.calmar), posNeg(m.calmar)],
    ["胜率", fmtPct(m.hit_ratio), ""],
    ["年化换手", m.ann_turnover == null ? "—" : fmtNum(m.ann_turnover, 0) + "×", ""],
    ["标的数", d.n_instruments, ""],
  ];
  $("#metric-strip").innerHTML = defs.map(([l, v, cls]) =>
    `<div class="metric"><div class="metric-label">${l}</div>
     <div class="metric-value ${cls}">${v}</div></div>`).join("");

  // 静默失效的两件事必须显式提示：杠杆上限长期绑定 = 波动目标没生效；
  // 换月识别不可信 = 这些标的的收益没被正确调整。
  const warns = [];
  const cap = d.leverage_cap_share;
  if (cap != null && cap > 0.5) {
    warns.push(`<div class="warn-bar"><b>波动目标未生效</b>
      杠杆上限 ${fmtNum(d.params.max_leverage, 0)}× 在 <b>${fmtPct(cap, 0)}</b> 的交易日绑定，
      目标 ${fmtPct(d.params.vol_target, 0)} 实际够不着（实现波动 ${fmtPct(m.ann_vol)}）。
      此时调"波动目标"滑块不会有反应——需要改 gross 或风险模型。</div>`);
  }
  if (d.roll_suspect && d.roll_suspect.length) {
    warns.push(`<div class="warn-bar info"><b>换月识别存疑 ${d.roll_suspect.length} 个标的</b>
      ${d.roll_suspect.join("、")}
      —— 持仓量/成交量信号识别不出合理的换月频率，这些标的的收益未被可靠的换月调整。</div>`);
  }
  $("#run-warnings").innerHTML = warns.join("");

  const eqOpt = lineOption(null, [
    { name: "费后", points: d.equity, color: C.amber },
    { name: "费前", points: d.gross_equity, color: C.faint },
  ]);
  eqOpt.legend = {
    top: 4, right: 12, icon: "roundRect", itemWidth: 14, itemHeight: 3,
    textStyle: { color: C.dim, fontSize: 10, fontFamily: MONO },
  };
  chart("ch-equity").setOption(eqOpt, true);

  chart("ch-dd").setOption(lineOption(null, [
    { name: "dd", points: d.drawdown, color: C.red }], { pct: true }), true);

  chart("ch-roll").setOption(lineOption(null, [
    { name: "sharpe", points: d.rolling_sharpe, color: C.cyan }], { area: false }), true);

  // 月度热力图
  const md = d.monthly;
  const heat = [];
  md.values.forEach((row, yi) => row.forEach((v, mi) => { if (v != null) heat.push([mi, yi, v]); }));
  const maxAbs = Math.max(0.001, ...heat.map(h => Math.abs(h[2])));
  chart("ch-monthly").setOption({
    grid: { left: 50, right: 14, top: 14, bottom: 52 },
    tooltip: { ...baseTooltip, formatter: p => `${md.years[p.value[1]]}-${md.months[p.value[0]]}  ${fmtPct(p.value[2])}` },
    xAxis: { type: "category", data: md.months.map(String), ...baseAxis },
    yAxis: { type: "category", data: md.years.map(String), ...baseAxis, inverse: true },
    visualMap: {
      min: -maxAbs, max: maxAbs, calculable: false, orient: "horizontal",
      left: "center", bottom: 2, itemWidth: 10, itemHeight: 80,
      textStyle: { color: C.dim, fontSize: 9.5, fontFamily: MONO },
      inRange: { color: [C.red, "#0a0a0a", C.green] },
    },
    series: [{
      type: "heatmap", data: heat,
      label: { show: true, fontSize: 8, color: "#b8b8b8", fontFamily: MONO, formatter: p => (p.value[2] * 100).toFixed(0) },
      itemStyle: { borderColor: "#000000", borderWidth: 1 },
    }],
  }, true);

  // 归因（首尾各 10）
  const at = d.attribution.length > 20
    ? [...d.attribution.slice(0, 10), ...d.attribution.slice(-10)]
    : d.attribution;
  chart("ch-attrib").setOption({
    grid: { left: 88, right: 44, top: 14, bottom: 26 },
    tooltip: { ...baseTooltip, formatter: p => `${p.name}  ${fmtPct2(p.value)}` },
    xAxis: { type: "value", ...baseAxis, axisLabel: { ...baseAxis.axisLabel, formatter: v => (v * 100).toFixed(1) + "%" } },
    yAxis: { type: "category", data: at.map(a => a.name), ...baseAxis, axisLabel: { ...baseAxis.axisLabel, fontSize: 9 } },
    series: [{
      type: "bar", barWidth: "60%",
      data: at.map(a => ({ value: a.contrib, itemStyle: { color: a.contrib >= 0 ? C.green + "cc" : C.red + "cc" } })),
    }],
  }, true);

  // 类别毛敞口
  const wbc = d.weights_by_class;
  chart("ch-classw").setOption({
    grid: { left: 48, right: 14, top: 34, bottom: 26 },
    tooltip: { ...baseTooltip, valueFormatter: v => fmtPct(v) },
    legend: { top: 4, textStyle: { color: C.dim, fontSize: 10, fontFamily: MONO }, icon: "roundRect", itemWidth: 12, itemHeight: 3 },
    xAxis: { type: "time", ...baseAxis },
    yAxis: { type: "value", ...baseAxis, axisLabel: { ...baseAxis.axisLabel, formatter: v => (v * 100).toFixed(0) + "%" } },
    series: wbc.series.map(s => ({
      name: s.name, type: "line", stack: "w", showSymbol: false,
      data: wbc.dates.map((dt, i) => [dt, s.data[i]]),
      lineStyle: { width: 1, color: CLASS_COLORS[s.name] || C.dim },
      itemStyle: { color: CLASS_COLORS[s.name] || C.dim },
      areaStyle: { opacity: 0.4 },
    })),
  }, true);

  // 最新持仓
  const lw = d.latest_weights;
  chart("ch-lastw").setOption({
    grid: { left: 88, right: 44, top: 14, bottom: 26 },
    tooltip: { ...baseTooltip, formatter: p => `${p.name}  ${fmtPct2(p.value)}` },
    xAxis: { type: "value", ...baseAxis, axisLabel: { ...baseAxis.axisLabel, formatter: v => (v * 100).toFixed(0) + "%" } },
    yAxis: { type: "category", data: lw.map(a => a.name).reverse(), ...baseAxis, axisLabel: { ...baseAxis.axisLabel, fontSize: 9 } },
    series: [{
      type: "bar", barWidth: "60%",
      data: lw.map(a => ({ value: a.weight, itemStyle: { color: a.weight >= 0 ? C.amber + "cc" : C.cyan + "cc" } })).reverse(),
    }],
  }, true);
}

/* ---------- 标的池页 ---------- */
let universeReady = false;
async function initUniverse() {
  if (universeReady) return;
  universeReady = true;
  if (!state.instruments) state.instruments = await api("/api/instruments");
  const inst = state.instruments;
  $("#universe-sub").textContent = `${inst.length} 个标的`;

  const classes = [...new Set(inst.map(i => i.asset_class))].sort(
    (a, b) => CLASS_ORDER.indexOf(a) - CLASS_ORDER.indexOf(b));
  $("#u-classes").innerHTML = classes.map(c =>
    `<button class="chip" data-value="${c}">${CLASS_LABELS[c] || c}</button>`).join("");
  $$("#u-classes .chip").forEach(ch => ch.addEventListener("click", () => {
    ch.classList.toggle("active"); renderUniverseTable();
  }));
  $("#u-search").addEventListener("input", renderUniverseTable);
  renderUniverseTable();
}

function pillFor(v) {
  if (v == null) return `<span class="pill flat">—</span>`;
  const cls = v > 0.0001 ? "pos" : v < -0.0001 ? "neg" : "flat";
  return `<span class="pill ${cls}">${fmtPct2(v)}</span>`;
}

async function renderUniverseTable() {
  const q = $("#u-search").value.trim().toUpperCase();
  const cls = new Set($$("#u-classes .chip.active").map(c => c.dataset.value));
  const rows = state.instruments.filter(i =>
    (!q || i.name.includes(q) || i.ric.toUpperCase().includes(q)) &&
    (!cls.size || cls.has(i.asset_class)));
  $("#u-tbody").innerHTML = rows.map(i => `
    <tr data-name="${i.name}">
      <td class="name-cell" style="color:var(--amber)">${i.name}</td>
      <td class="ric-cell">${i.ric}</td>
      <td><span class="pill class">${i.class_label}</span></td>
      <td class="num">${fmtPx(i.last)}${i.is_adjusted ? '<span class="adj-flag" title="涨跌幅与走势用换月调整后的价格指数，此处显示的是真实行情收盘价">adj</span>' : ""}</td>
      <td class="num">${pillFor(i.chg_1d)}</td>
      <td class="num">${pillFor(i.chg_1m)}</td>
      <td class="num">${pillFor(i.chg_1y)}</td>
      <td class="cell-spark" data-name="${i.name}"></td>
    </tr>`).join("");
  $$("#u-tbody tr").forEach(tr => tr.addEventListener("click", () => openDrawer(tr.dataset.name)));
  $$("#u-tbody .cell-spark").forEach(async td => {
    try {
      const d = await api(`/api/instruments/${td.dataset.name}/series?years=1`);
      td.innerHTML = sparkline(d.points.map(p => p[1]), { w: 120, h: 30, color: "auto" });
    } catch (_) { td.textContent = "—"; }
  });
}

/* ---------- 抽屉 ---------- */
async function openDrawer(name) {
  $("#drawer").classList.add("open");
  const meta = (state.instruments || []).find(i => i.name === name) || {};
  $("#drawer-title").textContent = name;
  $("#drawer-sub").textContent = `${meta.ric || ""} · ${meta.class_label || ""}`;
  try {
    const d = await api(`/api/instruments/${name}/series?years=5`);
    chart("ch-inst").setOption(lineOption(null, [
      { name, points: d.points, color: C.amber }]), true);
    setTimeout(() => chart("ch-inst").resize(), 300);
  } catch (e) {
    $("#ch-inst").innerHTML = `<div style="padding:24px;color:${C.red}">${e.message}</div>`;
  }
}
$("#drawer-close").addEventListener("click", () => $("#drawer").classList.remove("open"));
$("#drawer-mask").addEventListener("click", () => $("#drawer").classList.remove("open"));
document.addEventListener("keydown", e => { if (e.key === "Escape") $("#drawer").classList.remove("open"); });

/* ---------- 数据状态页 ---------- */
let dataReady = false;
async function initData() {
  if (dataReady) return;
  dataReady = true;
  const rows = await api("/api/data/status");
  const now = Date.now();
  $("#d-tbody").innerHTML = rows.map(r => {
    // 最新数据距今 >4 天（覆盖周末/假期）才算待更新
    const ageDays = r.last ? (now - Date.parse(r.last)) / 864e5 : Infinity;
    const stale = ageDays > 4;
    const rpy = r.rows_per_year == null ? "—" : fmtNum(r.rows_per_year, 0);
    return `<tr><td class="name-cell" style="color:var(--amber)">${r.ric}</td>
      <td class="num">${r.rows.toLocaleString()}</td>
      <td class="num ${r.sparse ? "neg" : ""}" title="${r.sparse ? "明显低于正常日线的 ~250 行/年，这条序列本身缺数" : ""}">${rpy}</td>
      <td class="num">${r.first || "—"}</td><td class="num">${r.last || "—"}</td>
      <td><span class="pill ${stale ? "stale" : "ok"}">${stale ? "待更新" : "最新"}</span></td></tr>`;
  }).join("");
}

/* ---------- 启动 ---------- */
startClock();
initOverview();
initTicker();
