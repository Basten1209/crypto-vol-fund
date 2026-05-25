const state = {
  data: null,
  dateIndex: 0,
  cycle: 14,
  policy: "daily_rebalance_to_target",
  playing: false,
  timer: null,
};

const els = {};

document.addEventListener("DOMContentLoaded", async () => {
  cacheElements();
  state.data = window.DASHBOARD_DATA || (await fetch("data/dashboard_snapshots.json").then((response) => response.json()));
  state.dateIndex = Math.max(0, state.data.dates.indexOf(state.data.metadata.default_date));
  state.cycle = state.data.metadata.default_cycle_days;
  state.policy = state.data.metadata.default_policy;
  initializeControls();
  render();
});

function cacheElements() {
  [
    "dateSlider",
    "playButton",
    "monthJumpButton",
    "selectedDate",
    "windowStatus",
    "aumInput",
    "kpiTotalReturn",
    "kpiTotalReturnHint",
    "kpiSharpe",
    "kpiMdd",
    "kpiVol",
    "kpiRisk",
    "kpiActive",
    "kpiTop",
    "performanceBadge",
    "equityChart",
    "drawdownChart",
    "monthlyChart",
    "portfolioWindow",
    "capBadge",
    "holdingsList",
    "orderSummary",
    "orderBadge",
    "ordersTable",
  ].forEach((id) => {
    els[id] = document.getElementById(id);
  });
}

function initializeControls() {
  els.dateSlider.max = String(state.data.dates.length - 1);
  els.dateSlider.value = String(state.dateIndex);
  els.dateSlider.addEventListener("input", () => {
    state.dateIndex = Number(els.dateSlider.value);
    render();
  });

  els.playButton.addEventListener("click", () => {
    state.playing ? stopDemo() : startDemo();
  });

  els.monthJumpButton.addEventListener("click", () => {
    jumpToNextRebalance();
  });

  els.aumInput.value = formatInteger(state.data.metadata.default_aum_krw);
  els.aumInput.addEventListener("input", () => {
    const raw = parseAum();
    els.aumInput.value = formatInteger(raw);
    renderOrders();
  });

  document.querySelectorAll("[data-cycle]").forEach((button) => {
    button.addEventListener("click", () => {
      state.cycle = Number(button.dataset.cycle);
      render();
    });
  });

  document.querySelectorAll("[data-policy]").forEach((button) => {
    button.addEventListener("click", () => {
      state.policy = button.dataset.policy;
      render();
    });
  });
}

function startDemo() {
  state.playing = true;
  els.playButton.textContent = "Ⅱ";
  state.timer = window.setInterval(() => {
    state.dateIndex = (state.dateIndex + 1) % state.data.dates.length;
    els.dateSlider.value = String(state.dateIndex);
    render();
  }, 650);
}

function stopDemo() {
  state.playing = false;
  els.playButton.textContent = "▶";
  window.clearInterval(state.timer);
  state.timer = null;
}

function jumpToNextRebalance() {
  const currentDate = selectedDate();
  const cycleSnapshot = snapshotForDate(currentDate).cycle;
  const next = cycleSnapshot.next_rebalance_date;
  if (!next) return;
  const nextIndex = state.data.dates.indexOf(next);
  if (nextIndex >= 0) {
    state.dateIndex = nextIndex;
    els.dateSlider.value = String(nextIndex);
    render();
  }
}

function render() {
  updateControlState();
  renderKpis();
  renderCharts();
  renderHoldings();
  renderOrders();
}

function updateControlState() {
  const date = selectedDate();
  const { cycle, policy } = snapshotForDate(date);
  const context = selectedWindowContext();
  els.selectedDate.textContent = date;
  els.windowStatus.textContent = `${policy.status} · ${policy.action}`;
  els.dateSlider.value = String(state.dateIndex);
  document.querySelectorAll("[data-cycle]").forEach((button) => {
    button.classList.toggle("is-active", Number(button.dataset.cycle) === state.cycle);
  });
  document.querySelectorAll("[data-policy]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.policy === state.policy);
  });
  els.performanceBadge.textContent = context.inHoldWindow
    ? `${state.cycle}D ${policy.label} · ${shortDate(context.holdStart)} to ${shortDate(context.selectedDate)}`
    : `${state.cycle}D ${policy.label} · Cash`;
  els.portfolioWindow.textContent = `${cycle.rebalance_date || "-"} target · hold ${policy.hold_start || "-"} to ${policy.hold_end || "-"}`;
}

function renderKpis() {
  const context = selectedWindowContext();
  const selected = context.policySnapshot;
  const mvWindow = windowSeries("minimum_variance", state.policy, state.cycle, context);
  const perf = performanceFromWindow(mvWindow);
  els.kpiTotalReturn.textContent = percent(perf?.total_return);
  els.kpiTotalReturnHint.textContent = context.inHoldWindow
    ? `${state.cycle}D ${selected.label} · ${shortDate(context.holdStart)} to ${shortDate(context.selectedDate)}`
    : "Off-window cash";
  els.kpiSharpe.textContent = decimal(perf?.sharpe_ratio, 3);
  els.kpiMdd.textContent = percent(perf?.max_drawdown);
  els.kpiVol.textContent = percent(perf?.annualized_volatility);
  els.kpiRisk.textContent = `Realized risk ${percent(perf?.realized_risk_annualized_mean)}`;
  els.kpiActive.textContent = selected.active_count ? String(selected.active_count) : "Cash";
  els.kpiTop.textContent = `Top weight ${percent(selected.top_weight)}`;
  els.capBadge.textContent = selected.top_weight <= state.data.metadata.single_asset_cap + 0.000001 ? "Cap OK" : "Cap breach";
  els.capBadge.classList.toggle("badge--ok", selected.top_weight <= state.data.metadata.single_asset_cap + 0.000001);
}

function renderCharts() {
  const context = selectedWindowContext();
  const mvSeries = windowSeries("minimum_variance", state.policy, state.cycle, context);
  const ewSeries = windowSeries("equal_weight", state.policy, state.cycle, context);
  const btcSeries = windowSeries("btc_hodl", "buy_and_hold", "btc", context);
  drawLineChart(els.equityChart, [
    { name: "Minimum variance", className: "line-mv", color: "var(--teal)", values: mvSeries.map((row) => row.equity) },
    { name: "Equal weight", className: "line-ew", color: "var(--coral)", values: ewSeries.map((row) => row.equity) },
    { name: "BTC", className: "line-btc", color: "var(--blue)", values: btcSeries.map((row) => row.equity) },
  ], context.holdStart, context.selectedDate);
  drawDrawdownChart(els.drawdownChart, mvSeries.map((row) => row.drawdown));
  drawMonthlyChart();
}

function renderHoldings() {
  const selected = snapshotForDate(selectedDate()).policy;
  if (!selected.holdings.length) {
    els.holdingsList.innerHTML = `<div class="empty">Cash/off-window 상태입니다. 다음 리밸런싱까지 포트폴리오 주문이 없습니다.</div>`;
    return;
  }
  els.holdingsList.innerHTML = selected.holdings
    .map((row) => {
      const width = Math.max(0.8, Math.min(100, row.weight * 100));
      return `
        <div class="holding-row">
          <span class="ticker">${escapeHtml(row.ticker)}</span>
          <span class="bar-track"><span class="bar-fill" style="width:${width}%"></span></span>
          <span class="weight">${percent(row.weight)}</span>
        </div>
      `;
    })
    .join("");
}

function renderOrders() {
  const selected = snapshotForDate(selectedDate()).policy;
  const entryAum = parseAum();
  const currentAum = entryAum * (selected.current_aum_multiplier ?? 1);
  const orderAum = entryAum * (selected.order_aum_multiplier ?? selected.current_aum_multiplier ?? 1);
  const materialOrders = selected.orders.filter((row) => Math.abs(row.delta_weight) > 0.00005);
  els.orderBadge.textContent = selected.action;
  els.orderSummary.textContent = `${selected.label} · turnover ${decimal(selected.order_abs_sum, 4)} · entry AUM ${krw(entryAum)} · current AUM ${krw(currentAum)}`;
  if (!materialOrders.length) {
    els.ordersTable.innerHTML = `<tr><td colspan="6" class="empty">No material order for this date.</td></tr>`;
    return;
  }
  els.ordersTable.innerHTML = materialOrders
    .slice(0, 30)
    .map((row) => {
      const sideClass = row.side === "BUY" ? "side--buy" : row.side === "SELL" ? "side--sell" : "side--hold";
      return `
        <tr>
          <td><strong>${escapeHtml(row.ticker)}</strong></td>
          <td><span class="side ${sideClass}">${row.side}</span></td>
          <td>${percent(row.current_weight)}</td>
          <td>${percent(row.target_weight)}</td>
          <td>${signedPercent(row.delta_weight)}</td>
          <td>${krw(row.delta_weight * orderAum)}</td>
        </tr>
      `;
    })
    .join("");
}

function drawLineChart(container, lines, startLabel = state.data.metadata.date_start, endLabel = state.data.metadata.date_end) {
  const { width, height } = chartSize(container, 640, 360);
  const pad = { top: 24, right: 26, bottom: 42, left: 54 };
  const allValues = lines.flatMap((line) => line.values).filter(Number.isFinite);
  if (!allValues.length) {
    drawEmptyChart(container, "No active hold window for this date.");
    return;
  }
  const minY = Math.min(...allValues, 0.7);
  const maxY = Math.max(...allValues, 1.35);
  const xMax = Math.max(...lines.map((line) => line.values.length - 1));
  const scaleX = (idx) => pad.left + (idx / Math.max(1, xMax)) * (width - pad.left - pad.right);
  const scaleY = (value) => pad.top + ((maxY - value) / (maxY - minY || 1)) * (height - pad.top - pad.bottom);
  const grid = [minY, 1, maxY]
    .map((value) => `<line class="grid-line" x1="${pad.left}" x2="${width - pad.right}" y1="${scaleY(value)}" y2="${scaleY(value)}"></line><text class="chart-label" x="8" y="${scaleY(value) + 4}">${decimal(value, 2)}</text>`)
    .join("");
  const paths = lines
    .map((line) => `<path class="${line.className}" d="${pathFromValues(line.values, scaleX, scaleY)}"></path>`)
    .join("");
  const markers = lines
    .map((line) => lineMarker(line, scaleX, scaleY))
    .join("");
  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}">
      ${grid}
      <line class="axis" x1="${pad.left}" x2="${width - pad.right}" y1="${height - pad.bottom}" y2="${height - pad.bottom}"></line>
      <line class="axis" x1="${pad.left}" x2="${pad.left}" y1="${pad.top}" y2="${height - pad.bottom}"></line>
      ${paths}
      ${markers}
      <text class="chart-label" x="${pad.left}" y="${height - 10}">${startLabel || "-"}</text>
      <text class="chart-label" x="${width - 108}" y="${height - 10}">${endLabel || "-"}</text>
    </svg>
  `;
}

function drawDrawdownChart(container, values) {
  const { width, height } = chartSize(container, 560, 360);
  const pad = { top: 24, right: 22, bottom: 40, left: 58 };
  if (!values.filter(Number.isFinite).length) {
    drawEmptyChart(container, "No active hold window for this date.");
    return;
  }
  const minY = Math.min(...values, -0.1);
  const maxY = 0;
  const scaleX = (idx) => pad.left + (idx / Math.max(1, values.length - 1)) * (width - pad.left - pad.right);
  const scaleY = (value) => pad.top + ((maxY - value) / (maxY - minY || 1)) * (height - pad.top - pad.bottom);
  const line = pathFromValues(values, scaleX, scaleY);
  const area = `${line} L ${width - pad.right} ${scaleY(0)} L ${pad.left} ${scaleY(0)} Z`;
  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}">
      <line class="grid-line" x1="${pad.left}" x2="${width - pad.right}" y1="${scaleY(0)}" y2="${scaleY(0)}"></line>
      <line class="grid-line" x1="${pad.left}" x2="${width - pad.right}" y1="${scaleY(minY)}" y2="${scaleY(minY)}"></line>
      <text class="chart-label" x="8" y="${scaleY(0) + 4}">0%</text>
      <text class="chart-label" x="8" y="${scaleY(minY) + 4}">${percent(minY)}</text>
      <path class="line-dd" d="${area}"></path>
      <path class="line-dd" d="${line}"></path>
      <circle class="chart-marker" cx="${scaleX(values.length - 1)}" cy="${scaleY(values[values.length - 1])}" r="3.5"></circle>
    </svg>
  `;
}

function drawMonthlyChart() {
  const context = selectedWindowContext();
  if (!context.inHoldWindow || !context.holdStart) {
    drawEmptyChart(els.monthlyChart, "Monthly history is hidden outside active hold windows.");
    return;
  }
  const cutoffMonth = context.holdStart.slice(0, 7);
  const rows = state.data.monthly_metrics
    .filter((row) => row.rebalance_policy === state.policy && Number(row.cycle_days) === state.cycle && row.month < cutoffMonth)
    .sort((a, b) => `${a.month}${a.strategy}`.localeCompare(`${b.month}${b.strategy}`));
  const byMonth = new Map();
  rows.forEach((row) => {
    if (!byMonth.has(row.month)) byMonth.set(row.month, {});
    byMonth.get(row.month)[row.strategy] = row.total_return;
  });
  state.data.monthly_returns
    .filter((row) => Number(row.cycle_days) === state.cycle && row.month < cutoffMonth)
    .forEach((row) => {
      if (!byMonth.has(row.month)) byMonth.set(row.month, {});
      byMonth.get(row.month).btc_hodl = row.btc_same_window_return;
    });
  const months = [...byMonth.keys()].sort();
  const values = months
    .flatMap((month) => [byMonth.get(month).minimum_variance, byMonth.get(month).equal_weight, byMonth.get(month).btc_hodl])
    .filter(Number.isFinite);
  if (!months.length || !values.length) {
    drawEmptyChart(els.monthlyChart, "No completed prior monthly hold windows yet.");
    return;
  }
  const { width, height } = chartSize(els.monthlyChart, 900, 260);
  const pad = { top: 24, right: 24, bottom: 56, left: 54 };
  const minY = Math.min(...values, -0.15);
  const maxY = Math.max(...values, 0.15);
  const zeroY = scaleLinear(0, minY, maxY, height - pad.bottom, pad.top);
  const groupWidth = (width - pad.left - pad.right) / Math.max(1, months.length);
  const barWidth = Math.max(6, groupWidth * 0.22);
  const bars = months
    .map((month, idx) => {
      const row = byMonth.get(month);
      const x = pad.left + idx * groupWidth + groupWidth * 0.18;
      return [
        bar(x, row.minimum_variance, minY, maxY, zeroY, height, pad, barWidth, "var(--teal)"),
        bar(x + barWidth + 4, row.equal_weight, minY, maxY, zeroY, height, pad, barWidth, "var(--coral)"),
        bar(x + (barWidth + 4) * 2, row.btc_hodl, minY, maxY, zeroY, height, pad, barWidth, "var(--blue)"),
        `<text class="chart-label" x="${x - 2}" y="${height - 16}" transform="rotate(-35 ${x - 2} ${height - 16})">${month.slice(2)}</text>`,
      ].join("");
    })
    .join("");
  els.monthlyChart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}">
      <line class="grid-line" x1="${pad.left}" x2="${width - pad.right}" y1="${zeroY}" y2="${zeroY}"></line>
      <text class="chart-label" x="8" y="${zeroY + 4}">0%</text>
      ${bars}
    </svg>
  `;
}

function bar(x, value, minY, maxY, zeroY, height, pad, barWidth, fill) {
  if (!Number.isFinite(value)) return "";
  const y = scaleLinear(value, minY, maxY, height - pad.bottom, pad.top);
  const top = Math.min(y, zeroY);
  const barHeight = Math.max(1, Math.abs(zeroY - y));
  return `<rect x="${x}" y="${top}" width="${barWidth}" height="${barHeight}" fill="${fill}" rx="2"></rect>`;
}

function lineMarker(line, scaleX, scaleY) {
  for (let idx = line.values.length - 1; idx >= 0; idx -= 1) {
    const value = line.values[idx];
    if (Number.isFinite(value)) {
      return `<circle class="chart-marker" cx="${scaleX(idx)}" cy="${scaleY(value)}" r="4" style="fill:${line.color || "currentColor"}"></circle>`;
    }
  }
  return "";
}

function drawEmptyChart(container, message) {
  container.innerHTML = `<div class="chart-empty">${escapeHtml(message)}</div>`;
}

function chartSize(container, fallbackWidth, fallbackHeight) {
  return {
    width: Math.max(320, Math.round(container.clientWidth || fallbackWidth)),
    height: Math.max(180, Math.round(container.clientHeight || fallbackHeight)),
  };
}

function pathFromValues(values, scaleX, scaleY) {
  return values
    .map((value, idx) => `${idx === 0 ? "M" : "L"} ${scaleX(idx).toFixed(2)} ${scaleY(value).toFixed(2)}`)
    .join(" ");
}

function scaleLinear(value, domainMin, domainMax, rangeMin, rangeMax) {
  return rangeMin + ((value - domainMin) / (domainMax - domainMin || 1)) * (rangeMax - rangeMin);
}

function selectedDate() {
  return state.data.dates[state.dateIndex];
}

function snapshotForDate(date) {
  const cycle = state.data.snapshots[date][String(state.cycle)];
  return { cycle, policy: cycle.policies[state.policy] };
}

function selectedWindowContext() {
  const date = selectedDate();
  const { cycle, policy } = snapshotForDate(date);
  return {
    selectedDate: date,
    holdStart: policy?.hold_start || null,
    holdEnd: policy?.hold_end || null,
    inHoldWindow: Boolean(policy?.in_hold_window),
    cycleSnapshot: cycle,
    policySnapshot: policy,
  };
}

function series(strategy, policy, cycle) {
  return state.data.daily_series[`${strategy}|${policy}|${cycle}`] || [];
}

function windowSeries(strategy, policy, cycle, context) {
  if (!context.inHoldWindow || !context.holdStart) return [];
  const rows = series(strategy, policy, cycle).filter((row) => {
    return row.date >= context.holdStart && row.date <= context.selectedDate;
  });
  let equity = 1;
  let peak = 1;
  return rows.map((row) => {
    equity *= 1 + row.daily_return;
    peak = Math.max(peak, equity);
    return {
      ...row,
      equity,
      drawdown: equity / peak - 1,
    };
  });
}

function performanceFromWindow(rows) {
  if (!rows.length) return null;
  const returns = rows.map((row) => row.daily_return).filter(Number.isFinite);
  const equity = rows.map((row) => row.equity).filter(Number.isFinite);
  const risks = rows.map((row) => row.realized_risk_annualized).filter(Number.isFinite);
  const volatility = returns.length >= 2 ? sampleStd(returns) * Math.sqrt(365) : Number.NaN;
  const returnStd = returns.length >= 2 ? sampleStd(returns) : Number.NaN;
  return {
    total_return: equity.length ? equity[equity.length - 1] - 1 : Number.NaN,
    annualized_volatility: volatility,
    sharpe_ratio: returns.length >= 2 && returnStd > 0 ? (mean(returns) / returnStd) * Math.sqrt(365) : Number.NaN,
    max_drawdown: equity.length ? Math.min(...localDrawdowns(equity)) : Number.NaN,
    realized_risk_annualized_mean: risks.length ? mean(risks) : Number.NaN,
  };
}

function localDrawdowns(equity) {
  let peak = Number.NEGATIVE_INFINITY;
  return equity.map((value) => {
    peak = Math.max(peak, value);
    return value / peak - 1;
  });
}

function mean(values) {
  if (!values.length) return Number.NaN;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function sampleStd(values) {
  if (values.length < 2) return Number.NaN;
  const avg = mean(values);
  const variance = values.reduce((total, value) => total + (value - avg) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

function shortDate(value) {
  return value ? value.slice(5) : "-";
}

function parseAum() {
  const value = Number(String(els.aumInput.value).replace(/[^0-9.-]/g, ""));
  return Number.isFinite(value) && value > 0 ? value : state.data.metadata.default_aum_krw;
}

function formatInteger(value) {
  return Math.round(value).toLocaleString("en-US");
}

function percent(value) {
  if (!Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(2)}%`;
}

function signedPercent(value) {
  if (!Number.isFinite(value)) return "-";
  const sign = value > 0 ? "+" : "";
  return `${sign}${percent(value)}`;
}

function decimal(value, places) {
  if (!Number.isFinite(value)) return "-";
  return value.toFixed(places);
}

function krw(value) {
  if (!Number.isFinite(value)) return "-";
  const sign = value < 0 ? "-" : "";
  return `${sign}₩${Math.abs(Math.round(value)).toLocaleString("en-US")}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
