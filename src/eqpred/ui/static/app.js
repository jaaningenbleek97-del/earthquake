const $ = (id) => document.getElementById(id);

const TABS = ["overview", "scoreboard", "ledger", "drivers", "lab"];
const ALIASES = { map: "overview", hunt: "lab", backtest: "lab", brain: "lab" };
const LAB_ANCHORS = { hunt: "lab-hunt", backtest: "lab-backtest", brain: "lab-brain" };
const FILTER_STORE = "eqpred.ledger.filters";
const MODE_STORE = "eqpred.leftover.mode";
const DRIVER_PLAIN = {
  etas_calibrated: "aftershock model",
  human: "a person",
};

const state = {
  snap: null,
  globeData: null,
  globe: null,
  globeSpin: true,
  offset: 0,
  charts: {},
  tab: "overview",
  selectedDay: "",
  selectedBox: "",
  leftoverMode: "all",
};

function formatDriver(item) {
  if (typeof item === "string") return DRIVER_PLAIN[item] || item;
  if (!item || !item.driver) return "";
  const name = DRIVER_PLAIN[item.driver] || item.driver;
  const bits = [name];
  if (item.lag_hours) bits.push(`${item.lag_hours}h`);
  const c = Number(item.contribution);
  if (Number.isFinite(c) && item.driver !== "etas_calibrated") {
    bits.push(`${c >= 0 ? "+" : ""}${c.toFixed(3)}`);
  }
  return bits.join(" ");
}

function formatDrivers(list) {
  return (list || []).map(formatDriver).filter(Boolean).join(", ");
}

function emptyRow(cols, text) {
  return `<tr><td colspan="${cols}" class="empty">${text}</td></tr>`;
}

function promoteLine(p) {
  const ms = state.snap?.mainshock;
  if (leftoverMode() === "lastweek" && ms?.last_week?.note) return ms.last_week.note;
  if (ms?.note) return ms.note;
  const held = state.globeData?.copy?.held ?? leftoverWeekHeld();
  if (held) return "Leftover cards show its attempt even when it loses. The historical rate is still the number we trust.";
  if (!p) return "The leftover model is trying to catch a fresh big quake. Aftershock rates are not the product.";
  if (p.pass) return "A leftover gate passed. The leftover call is still the experiment on the globe.";
  return "The leftover model is on the globe. Aftershock rates are the skeptic we refuse to ship.";
}

function leftoverMode() {
  return state.leftoverMode === "lastweek" ? "lastweek" : "all";
}

function leftoverWeekHeld() {
  const last = leftoverMode() === "lastweek";
  const week = last
    ? (state.snap?.mainshock?.last_week?.week || state.globeData?.last_week?.week || [])
    : (state.snap?.mainshock?.week || state.globeData?.week || []);
  const issued = week.filter((w) => ["global", "hikurangi", "alpine", "kermadec"].includes(w.region || w.region_name));
  if (!issued.length) {
    if (last) return Boolean(state.globeData?.last_week?.copy?.held ?? true);
    return Boolean(state.globeData?.copy?.held ?? true);
  }
  return issued.every((w) => w.held_to_climatology);
}

function leftoverView(card) {
  if (leftoverMode() === "lastweek" && card && card.last_week_p != null) {
    return {
      chance: card.last_week_chance_plain,
      sentence: card.last_week_sentence,
      why: card.last_week_why || [],
      held: card.last_week_held,
      climate: card.last_week_climatology_chance_plain || card.climatology_chance_plain,
    };
  }
  return {
    chance: card?.leftover_chance_plain || card?.chance_plain,
    sentence: card?.leftover_sentence || card?.sentence,
    why: card?.leftover_why || [],
    held: card?.held_to_climatology,
    climate: card?.climatology_chance_plain,
  };
}

function chanceKind(card) {
  const last = leftoverMode() === "lastweek";
  if (card?.held_to_climatology) {
    return last
      ? "last-week leftover attempt (does not beat the historical rate)"
      : "leftover attempt this week (does not beat the historical rate)";
  }
  return last ? "last-week leftover call" : "leftover call this week";
}

function syncModeButtons() {
  const last = leftoverMode() === "lastweek";
  document.querySelectorAll("[data-leftover-mode]").forEach((btn) => {
    const on = btn.dataset.leftoverMode === (last ? "lastweek" : "all");
    btn.classList.toggle("is-on", on);
    btn.setAttribute("aria-pressed", on ? "true" : "false");
  });
}

function setLeftoverMode(mode) {
  state.leftoverMode = mode === "lastweek" ? "lastweek" : "all";
  try {
    localStorage.setItem(MODE_STORE, state.leftoverMode);
  } catch (_) {
    /* ignore */
  }
  syncModeButtons();
  if (state.globeData) renderWeek(state.globeData);
  if (state.snap) renderMainshock(state.snap);
}

function activeCopy(data) {
  if (leftoverMode() === "lastweek") {
    return data?.last_week?.copy || state.globeData?.last_week?.copy || null;
  }
  return data?.copy || state.globeData?.copy || null;
}

function applyCopy(copy) {
  if (!copy) return;
  if (copy.lede && $("lede")) $("lede").textContent = copy.lede;
  if (copy.title) document.title = `${copy.title} — eqpred lab`;
}

function parseHash() {
  const raw = (location.hash || "#overview").replace(/^#/, "");
  const [name, qs] = raw.split("?");
  const tab = TABS.includes(name) ? name : (ALIASES[name] || "overview");
  const params = new URLSearchParams(qs || "");
  return {
    tab,
    source: name,
    filters: {
      model: params.get("model") || "",
      region: params.get("region") || "",
      kind: params.get("kind") || "",
      hit: params.get("hit") || "",
    },
  };
}

function ledgerFilters() {
  return {
    model: $("f-model")?.value || "",
    region: $("f-region")?.value || "",
    kind: $("f-kind")?.value || "",
    hit: $("f-hit")?.value || "",
  };
}

function applyLedgerFilters(filters) {
  if (!$("f-model") || !filters) return;
  if (filters.model != null && [...$("f-model").options].some((opt) => opt.value === filters.model)) {
    $("f-model").value = filters.model;
  }
  if (filters.region != null && [...$("f-region").options].some((opt) => opt.value === filters.region)) {
    $("f-region").value = filters.region;
  }
  if (filters.kind != null) $("f-kind").value = filters.kind;
  if (filters.hit != null) $("f-hit").value = filters.hit;
}

function storedFilters() {
  try {
    return JSON.parse(sessionStorage.getItem(FILTER_STORE) || "{}");
  } catch {
    return {};
  }
}

function persistFilters() {
  sessionStorage.setItem(FILTER_STORE, JSON.stringify(ledgerFilters()));
  writeHash(state.tab);
}

function writeHash(tab) {
  const filters = ledgerFilters();
  const q = new URLSearchParams();
  if (tab === "ledger") {
    Object.entries(filters).forEach(([key, value]) => {
      if (value) q.set(key, value);
    });
  }
  const next = q.toString() ? `#${tab}?${q}` : `#${tab}`;
  if (location.hash !== next) history.replaceState(null, "", next);
}

function coulombStat(payload) {
  const regions = payload?.regions || {};
  const on = Object.entries(regions)
    .filter(([, meta]) => meta && meta.used_on_issued)
    .map(([rid]) => rid);
  if (!on.length) return "tested, left off the published chance";
  return `on published chance (${on.join(", ")})`;
}

function migrationStat(payload) {
  const regions = payload?.regions || {};
  const on = Object.entries(regions)
    .filter(([, meta]) => meta && meta.used_on_issued)
    .map(([rid, meta]) => `${rid}: ${meta.column || "on"}`);
  if (!on.length) return "tested, left off the published chance";
  return `on published chance (${on.join("; ")})`;
}

function wave2Stat(payload) {
  const regions = payload?.regions || {};
  const on = Object.entries(regions)
    .filter(([, meta]) => meta && meta.used_on_issued)
    .map(([rid, meta]) => {
      const cols = (meta.columns || [])
        .filter((c) => c && c.used_on_issued)
        .map((c) => c.column);
      return cols.length ? `${rid}: ${cols.join(", ")}` : rid;
    });
  if (!on.length) return "tested, left off the published chance";
  return `on published chance (${on.join("; ")})`;
}

function pct(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function num(value, digits = 2) {
  if (value == null || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

function chance(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const n = Math.round(Math.max(0, Math.min(1, Number(value))) * 100);
  if (n <= 0) return "less than 1 in 100";
  return `${n} in 100`;
}

function placeShort(name) {
  return {
    global: "Earth-wide",
    hikurangi: "Hikurangi",
    alpine: "Alpine Fault",
    alpine_fault: "Alpine Fault",
    kermadec: "Kermadec",
    japan_trench: "Japan (practice)",
    chile_trench: "Chile (practice)",
  }[name] || String(name || "").replaceAll("_", " ");
}

async function loadFreshHunt(snap) {
  if (snap?.hunt_fresh?.n_tests) return snap.hunt_fresh;
  try {
    const pack = await get("/api/hunt?fresh=1");
    if (pack?.n_tests) {
      snap.hunt_fresh = pack;
      return pack;
    }
  } catch (_) {
    /* leftover hunt file may not exist yet */
  }
  return snap.hunt;
}

async function get(url, tries = 3) {
  let last;
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`${url} ${res.status}`);
      return await res.json();
    } catch (err) {
      last = err;
      if (i + 1 < tries) await new Promise((r) => setTimeout(r, 400));
    }
  }
  throw last;
}

function showTab(name, source) {
  if (ALIASES[name]) {
    source = source || name;
    name = ALIASES[name];
  }
  if (!TABS.includes(name)) name = "overview";
  state.tab = name;
  document.querySelectorAll(".tab").forEach((btn) => {
    const on = btn.dataset.tab === name;
    btn.classList.toggle("is-on", on);
    btn.setAttribute("aria-selected", on ? "true" : "false");
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("is-on", panel.id === name);
  });
  writeHash(name);
  if (name === "overview") {
    requestAnimationFrame(() => {
      if (state.globeData) drawGlobe(state.globeData);
      resizeGlobe();
    });
  }
  if (name === "drivers" && !state.charts.spf) loadSeries();
  if (name === "lab" && source && LAB_ANCHORS[source]) {
    $(LAB_ANCHORS[source])?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderRegistry(reg) {
  const el = $("registry-out");
  if (!el) return;
  if (!reg || !reg.id) {
    el.innerHTML = "<p class=\"hint\">No version pinned yet.</p>";
    return;
  }
  const hash = (value) => (value ? `${String(value).slice(0, 12)}…` : "—");
  el.innerHTML = `<div class="watch-list">
    <div class="watch-row"><span class="src">version</span><span class="pill fresh">${reg.id}</span></div>
    <div class="watch-row"><span class="src">created</span><span class="lag">${reg.created_at || "—"}</span></div>
    <div class="watch-row"><span class="src">model</span><span class="lag">${hash(reg.brain_hash)}</span></div>
    <div class="watch-row"><span class="src">code</span><span class="lag">${hash(reg.code_hash)}</span></div>
    <div class="watch-row"><span class="src">recipe</span><span class="lag">${reg.n_columns ?? "—"} columns · trained through ${reg.train_end || "—"}</span></div>
    <div class="watch-row"><span class="src">kept versions</span><span class="lag">${reg.n_versions ?? 1}</span></div>
  </div>`;
}

function huntRow(row) {
  const family = row.aftershock_shaped ? "catalog leftover" : row.family;
  const pill = row.aftershock_shaped ? "miss" : (row.confirmed ? "fresh" : "open");
  return `<tr class="hunt-row" data-driver="${row.driver}" data-lag="${row.lag_days}" data-region="${row.region}">
    <td><strong>${row.driver}</strong><div class="hint">lag ${row.lag_days}d · ${row.region} M≥${row.magnitude_min} · click to replay</div></td>
    <td><span class="pill ${pill}">${family}</span></td>
    <td>${num(row.train_rho, 3)}<div class="hint">q ${num(row.train_q, 3)}${row.train_lift != null ? ` · split lift ${num(row.train_lift, 2)}` : ""}</div></td>
    <td>${num(row.confirm_rho ?? row.holdout_rho, 3)}<div class="hint">late train p ${num(row.confirm_p ?? row.holdout_p, 3)}${row.replay_rho != null ? ` · replay ${num(row.replay_rho, 3)}` : ""}</div></td>
  </tr>`;
}

function renderHunt(hunt) {
  const summary = $("hunt-summary");
  const confirmedEl = $("hunt-confirmed");
  const loudEl = $("hunt-loudest");
  if (!summary || !confirmedEl || !loudEl) return;
  if (!hunt) {
    summary.innerHTML = "<p class=\"empty\">No hunt on disk yet.</p>";
    confirmedEl.innerHTML = `<thead><tr><th>Signal</th><th>Family</th><th>Old years</th><th>Later years</th></tr></thead><tbody>${emptyRow(4, "No hunt yet.")}</tbody>`;
    loudEl.innerHTML = `<thead><tr><th>Signal</th><th>Family</th><th>Old years</th><th>Later years</th></tr></thead><tbody>${emptyRow(4, "No hunt yet.")}</tbody>`;
    return;
  }
  const families = Object.entries(hunt.family_counts || {})
    .map(([name, n]) => `${name} ${n}`)
    .join(" · ");
  summary.innerHTML = `<div class="watch-list">
    <div class="watch-row"><span class="src">signals</span><span class="lag">${hunt.n_knobs ?? "—"} columns</span></div>
    <div class="watch-row"><span class="src">tests</span><span class="lag">${hunt.n_tests ?? "—"} (signal × lag × place)</span></div>
    <div class="watch-row"><span class="src">looked real on old years</span><span class="lag">${hunt.n_discovered ?? 0}</span></div>
    <div class="watch-row"><span class="src">same sign on late train</span><span class="pill fresh">${hunt.n_confirmed ?? 0}</span></div>
    <div class="watch-row"><span class="src">still true after train cut</span><span class="pill ${(hunt.n_replayed || 0) ? "fresh" : "open"}">${hunt.n_replayed ?? 0}</span></div>
    <div class="watch-row"><span class="src">not just aftershocks</span><span class="pill ${(hunt.n_confirmed_unconventional || 0) ? "fresh" : "open"}">${hunt.n_confirmed_unconventional ?? 0}</span></div>
    ${hunt.fresh ? `<div class="watch-row"><span class="src">target</span><span class="pill fresh">fresh mainshock</span></div>` : ""}
    <div class="watch-row"><span class="src">families</span><span class="lag">${families || "—"}</span></div>
  </div>`;
  const confirmed = (hunt.confirmed || []).slice(0, 40);
  confirmedEl.innerHTML = `<thead><tr><th>Signal</th><th>Family</th><th>Old years</th><th>Later years</th></tr></thead><tbody>${
    confirmed.length ? confirmed.map(huntRow).join("") : emptyRow(4, "Nothing survived the later-year replay.")
  }</tbody>`;
  loudEl.innerHTML = `<thead><tr><th>Signal</th><th>Family</th><th>Old years</th><th>Later years</th></tr></thead><tbody>${
    (hunt.loudest_train || []).map(huntRow).join("")
  }</tbody>`;
  document.querySelectorAll(".hunt-row").forEach((row) => {
    row.addEventListener("click", () => {
      $("bt-driver").value = row.dataset.driver;
      $("bt-lag").value = String(row.dataset.lag);
      $("bt-region").value = row.dataset.region;
      showTab("lab", "backtest");
      runBacktest();
    });
  });
}

function renderBacktestPack(pack) {
  const el = $("bt-pack");
  if (!el) return;
  const trades = pack?.trades || [];
  if (!trades.length) {
    el.innerHTML = `<thead><tr><th>Signal</th><th>On</th><th>Hits</th><th>Vs always-on</th></tr></thead><tbody>${emptyRow(4, "Replay one signal, or run a hunt first.")}</tbody>`;
    return;
  }
  el.innerHTML = `<thead><tr><th>Signal</th><th>On</th><th>Hits</th><th>Vs always-on</th></tr></thead><tbody>${
    trades.map((t) => `<tr class="hunt-row" data-driver="${t.driver}" data-lag="${t.lag_days}" data-region="${t.region}">
      <td><strong>${t.driver}</strong><div class="hint">lag ${t.lag_days}d · ${t.region_label || t.region} M≥${t.magnitude_min}</div></td>
      <td>${pct(t.time_in_alarm)} · ${t.n_trades} mornings</td>
      <td>${t.hit_rate == null ? "—" : pct(t.hit_rate)}</td>
      <td>base ${pct(t.base_rate)} · lift ${t.lift == null ? "—" : `${t.lift >= 0 ? "+" : ""}${(t.lift * 100).toFixed(1)} pt`}</td>
    </tr>`).join("")
  }</tbody>`;
  el.querySelectorAll(".hunt-row").forEach((row) => {
    row.addEventListener("click", () => {
      $("bt-driver").value = row.dataset.driver;
      $("bt-lag").value = String(row.dataset.lag);
      $("bt-region").value = row.dataset.region;
      runBacktest();
    });
  });
}

function renderTrade(trade) {
  const el = $("bt-out");
  if (!el) return;
  if (trade.error) {
    el.innerHTML = `<p class="empty">${trade.error}</p>`;
    return;
  }
  const lift = trade.lift == null ? "—" : `${trade.lift >= 0 ? "+" : ""}${(trade.lift * 100).toFixed(1)} pt`;
  const recent = (trade.recent || []).map((row) =>
    `<tr class="${row.hit ? "hit" : "miss"}"><td>${row.date}</td><td>${num(row.value, 3)}</td><td>${row.hit ? "happened" : "did not"}</td></tr>`
  ).join("");
  el.innerHTML = `<div class="grid stats">
    <div class="stat"><div class="k">mornings on</div><div class="v">${trade.n_trades}</div></div>
    <div class="stat"><div class="k">happened when on</div><div class="v">${trade.hit_rate == null ? "—" : pct(trade.hit_rate)}</div></div>
    <div class="stat"><div class="k">if always on</div><div class="v">${pct(trade.base_rate)}</div></div>
    <div class="stat"><div class="k">lift</div><div class="v">${lift}</div></div>
    <div class="stat"><div class="k">time on</div><div class="v">${pct(trade.time_in_alarm)}</div></div>
  </div>
  <p class="hint">${trade.note} Aimed at ${(trade.share * 100).toFixed(0)}% of old years; later years it was on ${pct(trade.time_in_alarm)}. Threshold ${num(trade.threshold, 3)} (${trade.side} tail) · ${trade.family} · ${trade.n_days} finished weeks.</p>
  ${recent ? `<div class="table-wrap"><table><thead><tr><th>Morning</th><th>Signal</th><th>Next 7 days</th></tr></thead><tbody>${recent}</tbody></table></div>` : ""}`;
}

async function runBacktest() {
  const driver = $("bt-driver").value;
  if (!driver) return;
  const q = new URLSearchParams({
    driver,
    lag: $("bt-lag").value,
    region: $("bt-region").value,
    share: $("bt-share").value,
  });
  $("bt-out").innerHTML = "<p class=\"hint\">Replaying finished weeks…</p>";
  try {
    const trade = await get(`/api/backtest?${q}`);
    if (trade.error) throw new Error(trade.error);
    renderTrade(trade);
  } catch (err) {
    $("bt-out").innerHTML = `<p class="empty">Replay failed: ${err.message}</p>`;
  }
}

function brainTradeRow(t) {
  return `<tr class="hunt-row" data-driver="${(t.legs && t.legs[0] && t.legs[0].driver) || t.driver || ""}" data-lag="${(t.legs && t.legs[0] && t.legs[0].lag_days) || t.lag_days || 0}" data-region="${t.region || "global"}">
    <td><strong>${t.driver}</strong><div class="hint">${t.kind || "single"} · ${t.region_label || t.region || ""}</div></td>
    <td>${t.n_trades ?? "—"} · ${pct(t.time_in_alarm)}</td>
    <td>${t.hit_rate == null ? "—" : pct(t.hit_rate)}</td>
    <td>base ${pct(t.base_rate)} · lift ${t.lift == null ? "—" : `${t.lift >= 0 ? "+" : ""}${(t.lift * 100).toFixed(1)} pt`}</td>
  </tr>`;
}

function wireBrainRows(table) {
  if (!table) return;
  table.querySelectorAll(".hunt-row").forEach((row) => {
    row.addEventListener("click", () => {
      if (!$("bt-driver") || !row.dataset.driver) return;
      $("bt-driver").value = row.dataset.driver;
      $("bt-lag").value = String(row.dataset.lag);
      $("bt-region").value = row.dataset.region;
      showTab("lab", "backtest");
      runBacktest();
    });
  });
}

function renderBrain(brain) {
  const note = $("brain-note");
  const stats = $("brain-stats");
  const conn = $("brain-conn");
  const exp = $("brain-exp");
  const journalEl = $("brain-journal");
  if (!note || !stats) return;
  const banner = $("brain-banner");
  const p = window.__promote;
  if (banner) {
    banner.textContent = promoteLine(p);
    banner.classList.toggle("ok", Boolean(p?.pass));
  }
  if (!brain || !brain.n_features) {
    note.textContent = brain?.note || "Press Learn again, or run the learn command.";
    stats.innerHTML = "";
    if (conn) conn.innerHTML = `<thead><tr><th>Connection</th><th>Family</th><th>Place</th><th>Weight</th></tr></thead><tbody>${emptyRow(4, "No learn on disk.")}</tbody>`;
    if (exp) exp.innerHTML = `<thead><tr><th>Try</th><th>On</th><th>Hits</th><th>Vs always-on</th></tr></thead><tbody>${emptyRow(4, "No experiments yet.")}</tbody>`;
    if (journalEl) journalEl.innerHTML = `<thead><tr><th>When</th><th>Lift</th><th>Leftover error</th><th>Walk-forward</th></tr></thead><tbody>${emptyRow(4, "No journal yet.")}</tbody>`;
    return;
  }
  note.textContent = brain.note || "";
  const lift = brain.locked_lift;
  const wf = brain.walk_forward || {};
  stats.innerHTML = [
    ["signals", brain.n_features],
    ["later lift", lift == null ? "—" : `${(lift * 100).toFixed(1)} pt`],
    ["leftover error", brain.locked_brier == null ? "—" : num(brain.locked_brier, 4)],
    ["aftershock-model error", brain.locked_etas_brier == null ? "—" : num(brain.locked_etas_brier, 4)],
    ["walk-forward lift", wf.mean_lift == null ? "—" : `${(wf.mean_lift * 100).toFixed(1)} pt`],
    ["journal", `${brain.journal_runs ?? 0} runs`],
    ["improving", brain.improved == null ? "first run" : brain.improved ? "yes" : "not this time"],
    ["quiet lift", brain.quiet?.locked_lift == null ? "—" : `${(brain.quiet.locked_lift * 100).toFixed(1)} pt`],
    ["published chance", "unchanged"],
  ].map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  const promoteEl = $("promote-note");
  if (promoteEl) {
    promoteEl.textContent = p?.reason ? (p.pass ? `Gate: ${p.reason}` : `Held because ${p.reason}`) : "";
  }
  const transferEl = $("transfer-note");
  if (transferEl && window.__transfer) {
    const t = window.__transfer;
    transferEl.textContent = t.n_pairs != null
      ? `Practice regions: ${t.n_beats_etas || 0}/${t.n_pairs} beat the aftershock model. Japan and Chile still do not get live cards.`
      : (t.note || "");
  }
  if (conn) {
    const connections = brain.connections || [];
    conn.innerHTML = `<thead><tr><th>Connection</th><th>Family</th><th>Place</th><th>Weight</th></tr></thead><tbody>${
      connections.length
        ? connections.map((c) => `<tr class="hunt-row" data-driver="${c.driver}" data-lag="${c.lag_days}" data-region="${c.region}">
          <td><strong>${c.driver}</strong><div class="hint">lag ${c.lag_days}d</div></td>
          <td><span class="pill">${c.family}</span></td>
          <td>${c.region_label || c.region}</td>
          <td>${num(c.importance, 3)}</td>
        </tr>`).join("")
        : emptyRow(4, "Nothing stood out.")
    }</tbody>`;
    wireBrainRows(conn);
  }
  if (exp) {
    const rows = brain.experiments || [];
    exp.innerHTML = `<thead><tr><th>Try</th><th>On</th><th>Hits</th><th>Vs always-on</th></tr></thead><tbody>${
      rows.length ? rows.map(brainTradeRow).join("") : emptyRow(4, "No experiments this run.")
    }</tbody>`;
    wireBrainRows(exp);
  }
  if (journalEl) {
    const rows = brain.journal || [];
    journalEl.innerHTML = `<thead><tr><th>When</th><th>Lift</th><th>Leftover error</th><th>Walk-forward</th></tr></thead><tbody>${
      rows.length
        ? rows.map((j) => `<tr>
          <td>${(j.ran_at || "").slice(0, 19)}</td>
          <td>${j.locked_lift == null ? "—" : `${(j.locked_lift * 100).toFixed(1)} pt`}</td>
          <td>${j.locked_brier == null ? "—" : num(j.locked_brier, 4)}</td>
          <td>${j.walk_mean_lift == null ? "—" : `${(j.walk_mean_lift * 100).toFixed(1)} pt`}${j.improved == null ? "" : j.improved ? " · better" : " · not better"}</td>
        </tr>`).join("")
        : emptyRow(4, "No journal yet.")
    }</tbody>`;
  }
}

async function fillBrain() {
  const pack = await get("/api/learn");
  window.__promote = pack.promote;
  window.__transfer = pack.transfer;
  renderBrain(pack.brain || pack);
}

async function retrainBrain() {
  const note = $("brain-note");
  if (note) note.textContent = "Learning on finished weeks…";
  try {
    const pack = await get("/api/learn?train=1");
    window.__promote = pack.promote;
    window.__transfer = pack.transfer;
    renderBrain(pack.brain || pack);
  } catch (err) {
    if (note) note.textContent = `Learn failed: ${err.message}`;
    const banner = $("brain-banner");
    if (banner) banner.textContent = `Learn failed: ${err.message}`;
  }
}

async function fillBacktestDrivers() {
  const select = $("bt-driver");
  if (!select) return;
  const meta = await get("/api/backtest");
  if (!select.options.length) {
    for (const name of meta.drivers || []) {
      select.insertAdjacentHTML("beforeend", `<option value="${name}">${name}</option>`);
    }
  }
  const regionSel = $("bt-region");
  if (regionSel && (meta.targets || []).length) {
    const kept = regionSel.value;
    regionSel.innerHTML = "";
    for (const t of meta.targets) {
      regionSel.insertAdjacentHTML(
        "beforeend",
        `<option value="${t.region}">${t.label || t.region} M≥${t.magnitude_min}</option>`
      );
    }
    if ([...regionSel.options].some((opt) => opt.value === kept)) regionSel.value = kept;
  }
  if (meta.pack) renderBacktestPack(meta.pack);
  const first = meta.pack && meta.pack.trades && meta.pack.trades[0];
  if (first && !select.dataset.seeded) {
    select.value = first.driver;
    $("bt-lag").value = String(first.lag_days ?? 0);
    $("bt-region").value = first.region;
    select.dataset.seeded = "1";
  }
}

function renderWatchdog(watch) {
  const el = $("watchdog-out");
  if (!el) return;
  if (!watch) {
    el.innerHTML = "<p class=\"hint\">No freshness report.</p>";
    return;
  }
  const blockers = (watch.blockers || []).join(", ") || "none";
  const verdict = watch.ok
    ? `<p>As of <code>${watch.as_of || "—"}</code> the required feeds are fresh. A card published this morning would stand.</p>`
    : `<p>As of <code>${watch.as_of || "—"}</code> we will not publish a new card: <code>${blockers}</code>. Existing cards stay as they are.</p>`;
  const rows = (watch.sources || []).map((src) => {
    const lag = src.lag_days == null ? "—" : `${Number(src.lag_days).toFixed(1)}d behind`;
    const avail = src.latest_available_at ? src.latest_available_at.slice(0, 10) : "no file";
    const req = src.required ? "required" : "optional";
    return `<div class="watch-row">
      <span class="src">${src.source}</span>
      <span class="pill ${src.status || "missing"}">${src.status || "missing"}</span>
      <span class="pill">${req}</span>
      <span class="lag">${avail} · ${lag}</span>
      <span class="role">${src.role || ""}</span>
    </div>`;
  }).join("");
  el.innerHTML = `${verdict}<div class="watch-list">${rows}</div>`;
}

function renderStatusBar(snap) {
  const el = $("status-bar");
  if (!el) return;
  const scored = snap.n_scored ?? snap.n_cards ?? 0;
  const open = snap.n_open_windows ?? 0;
  const watchOk = Boolean(snap.watchdog?.ok);
  const held = !snap.promote?.pass;
  el.innerHTML = `
    <span class="chip ${held ? "hold" : "ok"}"><strong>${held ? "Chance unchanged" : "Gate passed"}</strong></span>
    <span class="chip ${watchOk ? "ok" : "warn"}"><strong>${watchOk ? "Ready to publish" : "Too stale to publish"}</strong></span>
    <span class="chip"><strong>${scored}</strong> finished · ${open} waiting</span>
  `;
}

function boxPolygons(boxes) {
  const feats = [];
  for (const box of boxes || []) {
    for (const poly of box.polygons || []) {
      feats.push({
        ...box,
        geometry: {
          type: "Polygon",
          coordinates: [[
            [poly.min_lon, poly.min_lat],
            [poly.max_lon, poly.min_lat],
            [poly.max_lon, poly.max_lat],
            [poly.min_lon, poly.max_lat],
            [poly.min_lon, poly.min_lat],
          ]],
        },
      });
    }
  }
  return feats;
}

function magColor(mag) {
  if (mag >= 8) return "#f0c38a";
  if (mag >= 7) return "#e8a04a";
  if (mag >= 6) return "#d15b4a";
  return "#3d9a7a";
}

function resizeGlobe() {
  const el = $("globe-el");
  if (!el || !state.globe) return;
  state.globe.width(el.clientWidth);
  state.globe.height(el.clientHeight);
}

function lookAtNZ() {
  state.globe?.pointOfView({ lat: -40.5, lng: 174.5, altitude: 1.65 }, 900);
}

function lookAtEarth() {
  state.globe?.pointOfView({ lat: -20, lng: 170, altitude: 2.5 }, 900);
}

function setGlobeSpin(on) {
  state.globeSpin = on;
  if (state.globe?.controls) state.globe.controls().autoRotate = on;
  const btn = $("globe-spin");
  if (btn) btn.textContent = on ? "Pause spin" : "Spin the Earth";
}

function selectBox(id) {
  state.selectedBox = id || "";
  document.querySelectorAll(".call-card").forEach((card) => {
    card.classList.toggle("is-on", card.dataset.box === id);
  });
  const box = (state.globeData?.boxes || []).find((b) => b.id === id);
  if (box && state.globe) {
    state.globe.pointOfView({ lat: box.lat, lng: box.lon, altitude: 1.45 }, 800);
  }
  const hud = $("globe-hud");
  if (hud && box) {
    const card = box.card;
    const view = card && leftoverView(card);
    const hero = view?.chance || (card && (card.leftover_chance_plain || card.chance_plain || chance(card.value)));
    hud.innerHTML = card
      ? `<article class="call-card is-on"><p class="where">${box.label}</p><div class="chance">${hero}<small>${chanceKind({ held_to_climatology: view?.held })}</small></div></article>`
      : "";
  }
}

function heroChance(card) {
  const view = leftoverView(card);
  return view.chance || card.leftover_chance_plain || card.chance_plain || chance(card.value);
}

function renderWatch(items) {
  const el = $("watch-strip");
  if (!el) return;
  if (!items.length) {
    el.innerHTML = "";
    return;
  }
  el.innerHTML = `<p class="where">Leftover flags this morning</p><div class="flags">${items.map((item) =>
    `<span class="flag ${item.on ? "on" : ""}">${item.label}${item.on ? " on" : ""}</span>`
  ).join("")}</div>`;
}

function callCardHtml(card, { boxId, title, place, extra } = {}) {
  if (!card) return "";
  const anti = String(card.model_id || "").startsWith("anti.");
  const { cls, outcome } = outcomeBits(card);
  const view = leftoverView(card);
  const why = (view.why || []).join(", ");
  return `<article class="call-card${boxId === "global" ? " global" : ""}" data-box="${boxId || ""}" data-day="${(card.issued_at || "").slice(0, 10)}">
    <p class="where">${title || card.label || "Call"}</p>
    <h3>${place || card.place_plain || ""}</h3>
            <div class="chance">${anti ? (card.chance_plain || chance(card.value)) : heroChance(card)}<small>${anti ? "chance it stays quiet" : chanceKind({ held_to_climatology: view.held })}</small></div>
    <p class="sentence">${anti ? (card.sentence || "") : (view.sentence || card.sentence || "")}</p>
    <p class="etas">${card.mainshock
      ? `Constant guess from old years: ${view.climate || "—"} · aftershock model (too late): ${card.etas_chance_plain || chance(card.competitor_etas_p)}`
      : `Aftershock skeptic: ${card.etas_chance_plain || chance(card.competitor_etas_p)}${card.chance_plain && card.leftover_chance_plain && card.chance_plain !== card.leftover_chance_plain ? ` · catalog-only: ${card.chance_plain}` : ""}`
    } · <span class="pill ${cls}">${outcome}</span></p>
    ${why && !anti ? `<p class="why">Uses ${why}.</p>` : ""}
    ${extra || ""}
  </article>`;
}

function renderWeek(data) {
  state.globeData = data;
  applyCopy(activeCopy(data) || data.copy);
  renderWatch(data.watch || []);
  const globalEl = $("week-global");
  const boxesEl = $("week-boxes");
  const emptyEl = $("week-empty");
  if (!globalEl || !boxesEl) return;
  if (leftoverMode() === "lastweek" && !(data.last_week?.week || []).length) {
    globalEl.innerHTML = `<article class="call-card global"><p class="where">Last-week leftover</p><p class="sentence">This mode only looks at the last 7 days of leftover sources and calls next week. It has not been scored yet. Run <code>eqpred mainshock --last-week</code>. Official ledger cards stay calibrated ETAS.</p></article>`;
    boxesEl.innerHTML = "";
    if (emptyEl) emptyEl.hidden = true;
    return;
  }
  const g = data.global;
  globalEl.innerHTML = g
    ? callCardHtml(g, {
      boxId: "global",
      title: "Earth-wide",
      place: "Anywhere on Earth, magnitude 7+",
    })
    : "";
  const cards = (data.boxes || []).map((box) => {
    if (!box.card) return "";
    const extra = box.anti
      ? `<p class="sentence">${box.anti.sentence}</p>`
      : "";
    return callCardHtml(box.card, {
      boxId: box.id,
      title: `${box.label} · M≥${box.magnitude_min}`,
      place: {
        hikurangi: "Hikurangi trench, east of the North Island",
        alpine: "Alpine Fault, South Island",
        kermadec: "Kermadec trench, northeast of New Zealand",
      }[box.id] || box.place_short || box.label,
      extra,
    });
  }).join("");
  boxesEl.innerHTML = cards;
  if (emptyEl) emptyEl.hidden = Boolean(g || cards);
  const hud = $("globe-hud");
  if (hud && g) {
    hud.innerHTML = `<article class="call-card global"><p class="where">Earth-wide fresh big quake</p><div class="chance">${heroChance(g)}<small>magnitude 7+ this week</small></div></article>`;
  }
  document.querySelectorAll(".call-card").forEach((card) => {
    card.addEventListener("click", () => {
      const box = card.dataset.box;
      if (box && box !== "global") selectBox(box);
      if (card.dataset.day) openMorning(card.dataset.day, { drawer: false });
    });
  });
  const meta = $("globe-meta");
  if (meta) {
    const n = (data.quakes || []).length;
    const day = data.issue_day || "this week";
    meta.textContent = `Calls from ${day}. ${n} recent large quakes on the globe. Orange boxes are Hikurangi, the Alpine Fault, and Kermadec. Japan and Chile are practice regions only.`;
  }
  drawGlobe(data);
}

function drawGlobe(data) {
  const el = $("globe-el");
  if (!el) return;
  if (typeof Globe !== "function") {
    $("globe-meta").textContent = "The 3D globe could not load. This week’s chances are still listed beside it.";
    return;
  }
  const polygons = boxPolygons(data.boxes || []);
  const markers = (data.boxes || [])
    .filter((b) => b.card && Number.isFinite(b.lat) && Number.isFinite(b.lon))
    .map((b) => ({
      id: b.id,
      lat: b.lat,
      lng: b.lon,
      label: b.label,
      chance: heroChance(b.card),
    }));
  const quakes = (data.quakes || []).filter((q) => q.lat != null && q.lon != null);
  if (!state.globe) {
    try {
      state.globe = Globe()
        .globeImageUrl("https://cdn.jsdelivr.net/npm/three-globe@2.44.1/example/img/earth-blue-marble.jpg")
        .bumpImageUrl("https://cdn.jsdelivr.net/npm/three-globe@2.44.1/example/img/earth-topology.png")
        .backgroundColor("#07090d")
        .showAtmosphere(true)
        .atmosphereColor("#7eb6ff")
        .atmosphereAltitude(0.15)
        .polygonGeoJsonGeometry((d) => d.geometry)
        .polygonCapColor((d) => (d.id === state.selectedBox ? "rgba(232,160,74,0.35)" : "rgba(232,160,74,0.16)"))
        .polygonSideColor(() => "rgba(232,160,74,0.06)")
        .polygonStrokeColor(() => "#e8a04a")
        .polygonAltitude(0.01)
        .polygonLabel((d) => `${d.label}`)
        .onPolygonClick((d) => selectBox(d.id))
        .pointsData(quakes)
        .pointLat((d) => d.lat)
        .pointLng((d) => d.lon)
        .pointAltitude(0.008)
        .pointRadius((d) => Math.max(0.12, (Number(d.magnitude) - 4.8) * 0.11))
        .pointColor((d) => magColor(Number(d.magnitude)))
        .pointLabel((d) => `M${Number(d.magnitude).toFixed(1)} · ${d.place || ""}`)
        .htmlLat((d) => d.lat)
        .htmlLng((d) => d.lng)
        .htmlElement((d) => {
          const chip = document.createElement("div");
          chip.className = "globe-chip";
          chip.innerHTML = `<strong>${d.chance}</strong><span>${d.label}</span>`;
          chip.addEventListener("pointerdown", (ev) => {
            ev.stopPropagation();
            selectBox(d.id);
          });
          return chip;
        })
        (el);
      state.globe.controls().autoRotate = state.globeSpin;
      state.globe.controls().autoRotateSpeed = 0.38;
      lookAtNZ();
    } catch (err) {
      $("globe-meta").textContent = `The 3D globe failed (${err.message}). This week’s chances are still listed beside it.`;
      return;
    }
  }
  state.globe
    .polygonsData(polygons)
    .pointsData(quakes)
    .htmlElementsData(markers);
  resizeGlobe();
}

function renderOverview(snap) {
  if (state.globeData?.copy?.lede) applyCopy(state.globeData.copy);
  else $("lede").textContent = snap.what_plain || snap.what;
  const leftover = snap.brain?.locked_brier;
  const etasErr = snap.brain?.locked_etas_brier;
  const leftoverVs = leftover == null || etasErr == null
    ? "—"
    : `${num(leftover, 3)} / ${num(etasErr, 3)}`;
  const hero = [
    ["Finished weeks", String(snap.n_scored ?? snap.n_cards ?? "—")],
    ["Still waiting", String(snap.n_open_windows ?? "0")],
    ["Leftover / aftershock error", leftoverVs],
    ["Published chance", snap.promote?.pass ? "gate passed, still unchanged" : "unchanged"],
    ["This morning", snap.watchdog?.ok ? "feeds are fresh" : "too stale to publish"],
    ["Model version", snap.registry?.id ? String(snap.registry.id).slice(0, 17) : "unpinned"],
  ];
  if ($("stat-grid")) {
    $("stat-grid").innerHTML = hero
      .map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`)
      .join("");
  }
  const store = [
    ["Catalog events", snap.catalog.n_events?.toLocaleString() || "—"],
    ["M ≥ 8 in catalog", String(snap.catalog.n_m8 ?? "—")],
    ["Feature days", String(snap.features.n_days ?? "—")],
    ["Published cards", String(snap.n_cards ?? "—")],
    ["Hunt", snap.hunt?.n_tests ? `${snap.hunt.n_tests} tests · ${snap.hunt.n_confirmed_unconventional ?? 0} held` : "not run"],
    ["Fresh leftover hunt", snap.hunt_fresh?.n_confirmed != null ? `${snap.hunt_fresh.n_tests} tests · ${snap.hunt_fresh.n_confirmed} held` : "not run"],
    ["Where / M8 probe", snap.probe?.where_issued?.leftover_acc != null ? `${pct(snap.probe.where_issued.leftover_acc)} named the NZ box` : (snap.probe?.note ? "run" : "not run")],
    ["Replay", snap.backtest?.n_trades ? `${snap.backtest.n_trades} paper trades` : "replay finished weeks"],
    ["Learner", snap.brain?.n_features ? `${snap.brain.n_features} series · lift ${snap.brain.locked_lift == null ? "—" : `${(snap.brain.locked_lift * 100).toFixed(1)} pt`}` : "not run"],
    ["Practice transfer", snap.transfer?.n_pairs != null ? `${snap.transfer.n_beats_etas || 0}/${snap.transfer.n_pairs} beat the model` : "not run"],
    ["Train ends", snap.train_end],
    ["Later years start", snap.holdout_start],
    ["Catalog", `${snap.catalog.start || "?"} → ${snap.catalog.end || "?"}`],
    ["Quiet-week cut", num(snap.models?.quiet?.lambda_etas_max, 3)],
    ["How fast big quakes fade (b)", num(snap.models?.gr?.b, 3)],
    ["Earth-wide scale", num(snap.models?.calibration?.scale, 3)],
    ["Hikurangi scale", num(snap.models?.calibration?.scales?.hikurangi, 3)],
    ["Alpine scale", num(snap.models?.calibration?.scales?.alpine, 3)],
    ["Kermadec scale", num(snap.models?.calibration?.scales?.kermadec, 3)],
    ["Catalog nowcast", snap.models?.nowcast?.used_on_issued ? "on published chance" : "tested, left off"],
    ["Tidal stress", coulombStat(snap.models?.coulomb)],
    ["Storm cusp", coulombStat(snap.models?.cusp)],
    ["Wave 2 leftover", wave2Stat(snap.models?.wave2)],
    ["Box migration", migrationStat(snap.models?.migration)],
  ];
  const storeGrid = $("store-grid");
  if (storeGrid) {
    storeGrid.innerHTML = store
      .map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`)
      .join("");
  }
  const banner = $("promote-banner");
  if (banner) {
    banner.textContent = promoteLine(snap.promote);
    banner.classList.toggle("ok", Boolean(snap.promote?.pass));
  }
  renderStatusBar(snap);
  renderWatchdog(snap.watchdog);
  renderRegistry(snap.registry);
  renderHunt(snap.hunt_fresh || snap.hunt);
  renderBacktestPack(snap.backtest);
  window.__promote = snap.promote;
  window.__transfer = snap.transfer;
  renderBrain(snap.brain);

  const end = snap.features.end || snap.catalog.end;
  if (end) {
    if ($("day-input")) $("day-input").value = end.slice(0, 10);
    if ($("ser-end")) $("ser-end").value = end.slice(0, 10);
    const endDate = new Date(`${end}T00:00:00Z`);
    const serFrom = new Date(endDate);
    serFrom.setUTCFullYear(serFrom.getUTCFullYear() - 2);
    if ($("ser-start")) $("ser-start").value = serFrom.toISOString().slice(0, 10);
  }

  const select = $("f-model");
  if (select) {
    select.innerHTML = `<option value="">All</option>`;
    for (const [id, label] of Object.entries(snap.model_labels || {})) {
      select.insertAdjacentHTML("beforeend", `<option value="${id}">${label}</option>`);
    }
  }
  const regions = $("f-region");
  if (regions) {
    const kept = regions.value;
    regions.innerHTML = `<option value="">All</option>`;
    for (const region of Object.values(snap.regions || {})) {
      if (region.issued === false) continue;
      const name = region.name || region.label;
      const label = region.name === "global" ? "Earth-wide" : (region.label || name);
      regions.insertAdjacentHTML("beforeend", `<option value="${name}">${label}</option>`);
    }
    if ([...regions.options].some((opt) => opt.value === kept)) regions.value = kept;
  }

  const verdictLabel = {
    miss: "did not happen",
    hit_fresh: "fresh hit",
    hit_echo: "aftershock echo",
    hit_after_prior_m7: "hit after a prior M7",
    hit_mixed: "mixed",
  };
  const episodes = snap.episodes || [];
  $("episode-out").innerHTML = episodes.length
    ? episodes.map((ep) => {
      const cls = ep.verdict === "miss" ? "miss" : "hit";
      const quakes = (ep.window_m7 || []).map((q) =>
        `${(q.observed_at || "").slice(0, 10)} M${Number(q.magnitude).toFixed(1)} ${q.place || ""} ${q.aftershock_like ? "(echo)" : "(fresh)"}`
      ).join("<br>") || "No magnitude 7+ in the combined window.";
      const dates = (ep.issue_dates || []).map((d) =>
        `<button type="button" class="btn ghost day-link" data-day="${d}">${d}</button>`
      ).join(" ");
      return `<div class="episode">
        <div class="k">Episode ${ep.episode_id} · ${ep.start} → ${ep.end}</div>
        <p class="hint">${ep.n_cards} cards · ${ep.n_hits} happened · prior M7 in 14d: ${ep.prior_m7_14d} · ${formatDrivers(ep.drivers) || "—"}</p>
        <p><span class="pill ${cls}">${verdictLabel[ep.verdict] || ep.verdict}</span> fresh ${ep.n_fresh} · echo ${ep.n_echo} · aftershock-model chance ${chance(ep.etas_p)}</p>
        <p class="hint">${quakes}</p>
        <div class="row">${dates}</div>
      </div>`;
    }).join("")
    : `<p class="empty">No quiet-week alerts on the list.</p>`;
  document.querySelectorAll("#episode-out .day-link").forEach((btn) => {
    btn.addEventListener("click", () => openMorning(btn.dataset.day, { drawer: false }));
  });
}

function foreshockAttemptCardsHtml(pack) {
  const week = (pack?.week || []).filter((w) =>
    ["global", "hikurangi", "alpine", "kermadec"].includes(w.region)
  );
  if (!week.length) {
    return `<article class="card head-card leftover-week"><h3>This week’s catalog precursor</h3><p class="hint">Run <code>eqpred foreshock</code> to score small-quake counts known that morning against a fresh mainshock. Official ledger cards stay calibrated ETAS. This is not leftover.</p></article>`;
  }
  const cards = week.map((w) => {
    const raw = w.foreshock_raw_chance_plain || chance(w.foreshock_raw_p);
    const kind = w.held_to_climatology
      ? "catalog precursor attempt (does not beat the historical rate)"
      : "catalog precursor call this week";
    const why = (w.why || []).slice(0, 6).join(", ");
    return `<article class="call-card${w.region === "global" ? " global" : ""}">
      <p class="where">Catalog precursor · ${w.label} M≥${w.magnitude_min}</p>
      <div class="chance">${raw || "—"}<small>${kind}</small></div>
      <p class="sentence">${w.sentence || ""}</p>
      <p class="etas">Historical rate: ${w.climatology_chance_plain || "—"}</p>
      ${why ? `<p class="why">Uses ${why}.</p>` : ""}
    </article>`;
  }).join("");
  return `<article class="card head-card leftover-week"><h3>This week’s catalog precursor</h3><p class="hint">Small and moderate quakes known that morning, in a box that has not just had a large shock. Shown even when it loses. Official ledger cards stay calibrated ETAS. Not leftover. Not a public warning.</p><div class="grid heads leftover-attempt-grid">${cards}</div></article>`;
}

function leftoverAttemptCardsHtml(pack) {
  const last = leftoverMode() === "lastweek";
  const source = last ? (pack?.last_week || {}) : pack;
  const week = (source.week || []).filter((w) =>
    ["global", "hikurangi", "alpine", "kermadec"].includes(w.region)
  );
  if (!week.length) {
    return last
      ? `<article class="card head-card leftover-week"><h3>This week’s last-week leftover predictions</h3><p class="hint">This mode only looks at the last 7 days of leftover sources and calls next week. Run <code>eqpred mainshock --last-week</code> to score it. Official ledger cards stay calibrated ETAS.</p></article>`
      : "";
  }
  const cards = week.map((w) => {
    const raw = w.leftover_raw_chance_plain || chance(w.leftover_raw_p);
    const kind = w.held_to_climatology
      ? (last ? "last-week leftover attempt (does not beat the historical rate)" : "leftover attempt this week (does not beat the historical rate)")
      : (last ? "last-week leftover call" : "leftover call this week");
    const why = (w.why || []).slice(0, 6).join(", ");
    return `<article class="call-card${w.region === "global" ? " global" : ""}">
      <p class="where">${last ? "Last-week leftover" : "Leftover attempt"} · ${w.label} M≥${w.magnitude_min}</p>
      <div class="chance">${raw || "—"}<small>${kind}</small></div>
      <p class="sentence">${w.sentence || ""}</p>
      <p class="etas">Historical rate: ${w.climatology_chance_plain || "—"}</p>
      ${why ? `<p class="why">Uses ${why}.</p>` : ""}
    </article>`;
  }).join("");
  return `<article class="card head-card leftover-week"><h3>${last ? "This week’s last-week leftover predictions" : "This week’s leftover predictions"}</h3><p class="hint">${last ? "Only the past 7 days of leftover sources, calling next week. " : ""}Shown even when leftover loses. Official ledger cards stay calibrated ETAS. Not a public warning.</p><div class="grid heads leftover-attempt-grid">${cards}</div></article>`;
}

function renderMainshock(snap) {
  const pack = snap.mainshock;
  const note = $("mainshock-note");
  const grid = $("mainshock-heads");
  if (!grid) return;
  if (note) {
    note.textContent = leftoverMode() === "lastweek"
      ? (pack?.last_week?.note || pack?.note || "")
      : (pack?.note || "");
  }
  if (!pack?.regions?.length) {
    grid.innerHTML = `<p class="empty">Run <code>eqpred mainshock</code> to backtest the leftover model on fresh big quakes.</p>`;
    return;
  }
  const leftoverWeek = leftoverAttemptCardsHtml(pack);
  const regions = leftoverMode() === "lastweek" && pack.last_week?.regions?.length
    ? pack.last_week.regions
    : pack.regions;
  grid.innerHTML = leftoverWeek + regions.map((r) => {
    const beat = r.beats_climatology === true;
    const leaked = r.recipe_leaked;
    const verdict = beat ? " · leftover ranks and errors better" : leaked ? " · hunt leakage, does not count" : " · leftover does not win yet";
    const why = (r.why || []).slice(0, 4).join(", ");
    const tw = r.this_week || {};
    const attempt = tw.leftover_raw_chance_plain || chance(tw.leftover_raw_p);
    const attemptLine = attempt
      ? `<div class="k">this week's leftover attempt</div><div class="v">${attempt}${tw.held_to_climatology ? " · does not beat the historical rate" : ""}${tw.climatology_chance_plain ? ` · historical rate ${tw.climatology_chance_plain}` : ""}</div>`
      : "";
    return `<article class="card head-card">
      <h3>${r.label} · M≥${r.magnitude_min}</h3>
      <p class="hint">${r.n_score} later weeks · fresh hit rate ${pct(r.fresh_rate_holdout)} (aftershocks included: ${pct(r.raw_rate_holdout)})</p>
      ${attemptLine}
      <div class="k">error vs a constant guess</div><div class="v">${num(r.brier, 3)} / ${num(r.climatology_brier, 3)}${verdict}</div>
      <div class="k">recipe</div><div class="v">${r.recipe || "—"}</div>
      <div class="k">when leftover was loudest</div><div class="v">${pct(r.hit_rate)} happened · ${pct(r.time_in_alarm)} of days · lift ${r.lift == null ? "—" : `${(r.lift * 100).toFixed(1)} pt`}</div>
      ${why ? `<p class="hint">Leaned on ${why}.</p>` : ""}
    </article>`;
  }).join("");
  const extras = leftoverMode() === "lastweek" ? [] : (pack.experiments || []);
  if (extras.length) {
    grid.insertAdjacentHTML("beforeend", extras.map((exp) => {
      const title = exp.id === "fresh_30d"
        ? "Next 30 days — leftover vs a constant"
        : exp.id === "fresh_7d_m8"
          ? "Earth-wide M8 next week — leftover vs a constant"
          : "Quiet-start next week — leftover vs a constant";
      const rows = (exp.regions || []).map((r) => {
        const beat = r.beats_climatology === true;
        const held = r.this_week?.held_to_climatology;
        const lift = r.lift == null ? "" : ` · lift ${(r.lift * 100).toFixed(1)} pt`;
        const holdTag = held ? " · this week held" : "";
        return `<div class="watch-row"><span class="src">${r.label} M≥${r.magnitude_min}</span><span class="pill ${beat ? "fresh" : "open"}">${beat ? "ranks better" : "constant still wins"} · ${num(r.brier, 3)} / ${num(r.climatology_brier, 3)}${lift}${holdTag}</span></div>`;
      }).join("") || "<p class=\"empty\">Not enough later weeks to score.</p>";
      return `<article class="card head-card"><h3>${title}</h3><p class="hint">${exp.n_beats_climatology || 0} of ${(exp.regions || []).length} beat a constant guess.</p><div class="watch-list">${rows}</div></article>`;
    }).join(""));
  }
  renderProbe(snap.probe);
  renderForeshock(snap);
}

function renderForeshock(snap) {
  const pack = snap.foreshock;
  const note = $("foreshock-note");
  const grid = $("foreshock-heads");
  if (!grid) return;
  if (note && pack?.note) note.textContent = pack.note;
  if (!pack?.regions?.length) {
    grid.innerHTML = `<p class="empty">Run <code>eqpred foreshock</code> to backtest the catalog precursor on fresh big quakes.</p>`;
    return;
  }
  const week = foreshockAttemptCardsHtml(pack);
  grid.innerHTML = week + (pack.regions || []).map((r) => {
    const beat = r.beats_climatology === true;
    const verdict = beat ? " · catalog precursor ranks and errors better" : " · catalog precursor does not win yet";
    const why = (r.why || []).slice(0, 4).join(", ");
    const tw = r.this_week || {};
    const attempt = tw.foreshock_raw_chance_plain || chance(tw.foreshock_raw_p);
    const attemptLine = attempt
      ? `<div class="k">this week's catalog precursor attempt</div><div class="v">${attempt}${tw.held_to_climatology ? " · does not beat the historical rate" : ""}${tw.climatology_chance_plain ? ` · historical rate ${tw.climatology_chance_plain}` : ""}</div>`
      : "";
    return `<article class="card head-card">
      <h3>${r.label} · M≥${r.magnitude_min}</h3>
      <p class="hint">${r.n_score} later quiet-start weeks · fresh hit rate ${pct(r.fresh_rate_holdout)} (aftershocks included: ${pct(r.raw_rate_holdout)})</p>
      ${attemptLine}
      <div class="k">error vs a constant guess</div><div class="v">${num(r.brier, 3)} / ${num(r.climatology_brier, 3)}${verdict}</div>
      <div class="k">recipe</div><div class="v">${r.recipe || "—"}</div>
      <div class="k">when the precursor was loudest</div><div class="v">${pct(r.hit_rate)} happened · ${pct(r.time_in_alarm)} of days · lift ${r.lift == null ? "—" : `${(r.lift * 100).toFixed(1)} pt`}</div>
      ${why ? `<p class="hint">Leaned on ${why}.</p>` : ""}
    </article>`;
  }).join("");
}

function renderProbe(pack) {
  const grid = $("mainshock-heads");
  if (!grid || !pack) return;
  const m8 = pack.m8 || {};
  const shifts = (m8.shifts || []).map((row) => {
    const collapsed = (row.time_in_alarm || 0) >= 0.99;
    const lift = collapsed ? "collapsed to a constant" : (row.lift == null ? "—" : `lift ${(row.lift * 100).toFixed(1)} pt`);
    return `<div class="watch-row"><span class="src">${row.shift_days ? `leftover shifted ${row.shift_days}d` : "aligned leftover"}</span><span class="pill ${row.shift_days ? "open" : "fresh"}">${lift}${collapsed ? "" : ` · ${row.n_hits ?? 0}/${row.n_trades ?? 0} loud hits`}</span></div>`;
  }).join("");
  const leak = m8.looks_like_leakage
    ? "Shifted leftover still ranks — treat the M8 hint as season or leakage."
    : "A leftover shift did not keep the same ranking.";
  const whereCard = (w, title) => {
    if (!w) return "";
    if (w.leftover_acc == null) {
      return `<article class="card head-card"><h3>${title}</h3><p class="hint">${w.note || "Not enough later weeks."}</p></article>`;
    }
    const beat = w.beats_majority;
    return `<article class="card head-card">
      <h3>${title}</h3>
      <p class="hint">${w.n_hold} later mornings with a fresh hit in exactly one box.</p>
      <div class="k">leftover named the box</div><div class="v">${pct(w.leftover_acc)} · always-guess ${w.majority_label || w.majority} ${pct(w.majority_acc)}${beat ? " · leftover wins" : " · leftover does not win"}</div>
    </article>`;
  };
  grid.insertAdjacentHTML("beforeend", `
    <article class="card head-card">
      <h3>Is the M8 ranking real?</h3>
      <p class="hint">${pack.note || leak}</p>
      <div class="watch-list">${shifts || '<p class="empty">Run <code>eqpred probe</code>.</p>'}</div>
    </article>
    ${whereCard(pack.where_issued, "Which NZ box — leftover vs always-guess")}
    ${whereCard(pack.where_transfer, "Japan vs Chile (practice) — leftover vs always-guess")}
  `);
}

function renderScoreboard(snap) {
  renderMainshock(snap);
  $("head-cards").innerHTML = (snap.heads || [])
    .map((h) => {
      const tia = h.time_in_alarm == null ? "" : `<div class="k">share of days on alert</div><div class="v">${pct(h.time_in_alarm)}</div>`;
      const month = h.max_days_in_month == null ? "" : `<div class="k">alert days / month</div><div class="v">${num(h.mean_days_per_month, 1)} typical · ${h.max_days_in_month} max${h.budget_days_per_month == null ? "" : ` / cap ${h.budget_days_per_month}`}</div>`;
      const etas = h.mean_etas_p == null ? "" : `<div class="k">aftershock-model chance</div><div class="v">${chance(h.mean_etas_p)}</div>`;
      const brier = h.brier == null ? "" : `<div class="k">error score (lower is better)</div><div class="v">${num(h.brier, 3)}${h.brier_etas == null ? "" : ` vs ${num(h.brier_etas, 3)}`}</div>`;
      const rank = h.p_hit_corr == null && h.resolution == null ? "" : `<div class="k">does a higher chance rank better weeks?</div><div class="v">${num(h.p_hit_corr, 3)}</div>`;
      const near = h.mean_near_miss == null ? "" : `<div class="k">how close on misses</div><div class="v">${num(h.mean_near_miss, 3)}</div>`;
      const quiet = h.quiet_base_rate == null ? "" : `<div class="k">vs always saying quiet</div><div class="v">${pct(h.hit_rate)} / ${pct(h.quiet_base_rate)}${h.quiet_lift == null ? "" : ` · ${(h.quiet_lift * 100).toFixed(1)} pt`}</div>`;
      const anti = String(h.model_id || "").startsWith("anti.");
      return `<article class="card head-card">
        <h3>${h.label}</h3>
        <p class="hint">${h.n} finished calls</p>
        <div class="k">${anti ? "quiet held" : "how often it happened"}</div><div class="v">${pct(h.hit_rate)}</div>
        ${quiet}
        <div class="k">typical published chance</div><div class="v">${chance(h.mean_p)}</div>
        ${etas}
        ${brier}
        ${rank}
        ${near}
        ${tia}
        ${month}
      </article>`;
    })
    .join("") || `<p class="empty">No finished calls yet.</p>`;

  const rows = (snap.battery || []).map((b) => {
    const skill = b.brier != null
      ? `${pct(b.hit_rate)} happened · typical chance ${chance(b.mean_p)}`
      : b.hit_rate == null ? "—" : `${pct(b.hit_rate)} happened · ${pct(b.time_in_alarm)} on alert`;
    const extra = b.brier != null
      ? `Error ${num(b.brier, 3)} / aftershock model ${num(b.brier_etas, 3)}${b.beats_etas ? " · beats the aftershock model" : ""}`
      : b.monte_carlo_p_ge_observed != null
      ? `p=${num(b.monte_carlo_p_ge_observed, 3)}`
      : b.bootstrap_p_ge_observed != null
        ? `p=${num(b.bootstrap_p_ge_observed, 3)}`
        : [b.corr_kp, b.corr_dst_nt, b.corr_sunspot_number].some((x) => x != null)
          ? `r(Kp)=${num(b.corr_kp, 3)} r(Dst)=${num(b.corr_dst_nt, 3)} r(SN)=${num(b.corr_sunspot_number, 3)}`
          : b.difference != null
            ? `Δ proton ${num(b.difference, 3)}`
            : "";
    return `<tr>
      <td><strong>${b.name}</strong><div class="hint">${b.blurb || ""}</div></td>
      <td>${skill}</td>
      <td>${extra}</td>
    </tr>`;
  });
  $("battery-table").innerHTML = `<thead><tr><th>Test</th><th>Later years</th><th>Detail</th></tr></thead><tbody>${rows.join("") || emptyRow(3, "No fairness tests on disk.")}</tbody>`;

  const stratRows = (snap.strategies || []).map((s) => {
    const tag = s.locked ? "locked" : (s.score_fresh ? "fresh paper" : "experiment");
    const useFresh = Boolean(s.score_fresh);
    const rate = useFresh ? s.fresh_hit_rate : s.hit_rate;
    const liftVal = useFresh ? s.fresh_lift : s.lift;
    const base = useFresh ? s.fresh_base_rate : s.base_rate;
    const hit = rate == null ? "—" : `${pct(rate)} of ${s.n_trades} times`;
    const lift = liftVal == null ? "—" : `${liftVal >= 0 ? "+" : ""}${(liftVal * 100).toFixed(1)} pt`;
    const extra = useFresh && s.hit_rate != null
      ? `<div class="hint">raw (incl. aftershocks) ${pct(s.hit_rate)} · ${s.fresh_n_hits ?? 0} fresh hits</div>`
      : "";
    return `<tr>
      <td><strong>${s.strategy_id}</strong><div class="hint">${s.label}<br><span class="pill ${s.locked ? "hit" : "miss"}">${tag}</span></div></td>
      <td>${hit}${extra}</td>
      <td>${pct(s.time_in_alarm)} on alert · base ${pct(base)} · lift ${lift}</td>
    </tr>`;
  });
  $("strategy-table").innerHTML = `<thead><tr><th>Rule</th><th>How often it happened</th><th>Vs always-on</th></tr></thead><tbody>${stratRows.join("") || emptyRow(3, "No rules on disk.")}</tbody>`;

  const relRows = [];
  for (const block of snap.reliability || []) {
    const name = `${(state.snap?.model_labels || {})[block.model_id] || block.model_id || ""} · ${placeShort(block.region)}`;
    for (const bin of block.bins || []) {
      relRows.push(`<tr>
        <td>${name}</td>
        <td>${chance(bin.lo)}–${chance(bin.hi)}</td>
        <td>${bin.n}</td>
        <td>${chance(bin.mean_p)}</td>
        <td>${pct(bin.hit_rate)}</td>
      </tr>`);
    }
  }
  $("reliability-table").innerHTML = `<thead><tr><th>Call type</th><th>We said</th><th>Weeks</th><th>Typical chance</th><th>It happened</th></tr></thead><tbody>${relRows.join("") || emptyRow(5, "No chance cards.")}</tbody>`;
}

function isOpenWindow(row) {
  if (row.window_closed === false) return true;
  return row.hit == null;
}

function outcomeBits(row) {
  const anti = String(row.model_id || "").startsWith("anti.");
  if (isOpenWindow(row)) return { cls: "open", outcome: "still waiting" };
  const score = Number(row.near_miss);
  const near = Number.isFinite(score) && score > 0 && score < 1;
  if (!anti && !row.hit && near) return { cls: "near", outcome: "close miss" };
  if (anti && !row.hit) return { cls: "miss", outcome: "broke" };
  const cls = row.hit ? "hit" : "miss";
  const outcome = anti ? (row.hit ? "stayed quiet" : "broke") : (row.hit ? "happened" : "did not");
  return { cls, outcome };
}

function nearHint(row) {
  if (isOpenWindow(row)) return "";
  const anti = String(row.model_id || "").startsWith("anti.");
  const score = Number(row.near_miss);
  if (!Number.isFinite(score) || score <= 0) return "";
  if (!anti && row.hit) return "";
  if (anti && row.hit && score >= 1) return "";
  const bits = [`close ${num(row.near_miss, 2)}`];
  if (row.miss_km != null && Number(row.miss_km) > 0) bits.push(`${num(row.miss_km, 0)} km away`);
  if (row.miss_hours != null && Number(row.miss_hours) > 0) bits.push(`${num(row.miss_hours, 0)} h late`);
  if (row.miss_mag != null && Number(row.miss_mag) > 0) bits.push(`M-${num(row.miss_mag, 1)}`);
  return ` <span class="hint">${bits.join(" · ")}</span>`;
}

async function loadLedger() {
  persistFilters();
  const params = new URLSearchParams({
    model: $("f-model").value,
    region: $("f-region").value,
    kind: $("f-kind").value,
    hit: $("f-hit").value,
    limit: "120",
    offset: String(state.offset),
  });
  $("ledger-meta").textContent = "Loading calls…";
  try {
    const data = await get(`/api/ledger?${params}`);
    const total = data.total || 0;
    const offset = data.offset || 0;
    const rows = data.rows || [];
    $("ledger-meta").textContent = total
      ? `${total} calls · showing ${offset + 1}–${Math.min(offset + rows.length, total)}`
      : "No calls match these filters.";
    const body = rows.map((r) => {
      const anti = String(r.model_id || "").startsWith("anti.");
      const { cls, outcome } = outcomeBits(r);
      const drivers = formatDrivers(r.drivers);
      const issued = (r.issued_at || "").slice(0, 10);
      const events = isOpenWindow(r) ? "" : ` ${r.n_events || 0}`;
      const on = issued && issued === state.selectedDay ? " is-on" : "";
      return `<tr class="ledger-row ${cls}${on}" data-day="${issued}">
        <td><button type="button" class="btn ghost day-link" data-day="${issued}">${issued}</button></td>
        <td>${r.label || r.model_id}</td>
        <td>${r.place_short || placeShort(r.region_name)} M≥${r.magnitude_min}</td>
        <td><span class="pill ${r.kind}">${anti ? "stays quiet" : "chance"}</span> ${r.chance_plain || chance(r.value)}</td>
        <td>${r.etas_chance_plain || (r.competitor_etas_p == null ? "—" : chance(r.competitor_etas_p))}</td>
        <td><span class="pill ${cls}">${outcome}</span>${events}${nearHint(r)}</td>
        <td>${drivers}</td>
      </tr>`;
    }).join("");
    $("ledger-table").innerHTML = `<thead><tr>
      <th>Morning</th><th>Kind</th><th>Place</th><th>We said</th><th>Aftershock model</th><th>What happened</th><th>Why</th>
    </tr></thead><tbody>${body || emptyRow(7, "No calls match these filters.")}</tbody>`;
    $("ledger-table").querySelectorAll(".ledger-row").forEach((row) => {
      row.addEventListener("click", (ev) => {
        if (ev.target.closest("a")) return;
        openMorning(row.dataset.day, { drawer: true });
      });
    });
  } catch (err) {
    $("ledger-meta").textContent = "";
    $("ledger-table").innerHTML = `<thead><tr>
      <th>Morning</th><th>Kind</th><th>Place</th><th>We said</th><th>Aftershock model</th><th>What happened</th><th>Why</th>
    </tr></thead><tbody>${emptyRow(7, `List failed: ${err.message}`)}</tbody>`;
  }
}

function flagList(features) {
  if (!features) return "";
  const names = [
    ["spf_significant_window", "solar proton window"],
    ["proton_spike", "proton spike"],
    ["kp_high", "magnetic storm (Kp)"],
    ["dst_storm", "magnetic storm (Dst)"],
    ["near_new_or_full", "new or full moon"],
  ];
  return `<div class="flags">${names.map(([k, label]) => {
    const on = Number(features[k] || 0) > 0.5;
    return `<span class="flag ${on ? "on" : ""}">${label}${on ? " on" : ""}</span>`;
  }).join("")}</div>
  <p class="hint">These leftover signals were known that morning. They are in our experimental call.</p>`;
}

function dayHtml(data) {
  const cards = (data.cards || []).map((c) => {
    const anti = String(c.model_id || "").startsWith("anti.");
    const { cls, outcome } = outcomeBits(c);
    return `<tr class="${cls}">
      <td>${c.label || c.model_id}</td>
      <td>${c.place_short || placeShort(c.region_name)} M≥${c.magnitude_min}</td>
      <td>${c.leftover_sentence || c.sentence || `${anti ? "stays quiet" : "chance"} ${heroChance(c)}`}</td>
      <td>${c.etas_chance_plain || (c.competitor_etas_p == null ? "—" : chance(c.competitor_etas_p))}</td>
      <td><span class="pill ${cls}">${outcome}</span>${nearHint(c)}</td>
    </tr>`;
  }).join("");
  const quakes = (data.quakes || []).map((q) =>
    `<tr><td>${(q.observed_at || "").replace("T", " ").slice(0, 16)}</td><td>M ${num(q.magnitude, 1)}</td><td>${q.place || ""}</td></tr>`
  ).join("");
  return `
    ${flagList(data.features)}
    <div class="table-wrap"><table><thead><tr><th>Kind</th><th>Place</th><th>We said</th><th>Aftershock model</th><th>What happened</th></tr></thead><tbody>${cards || emptyRow(5, "No calls that morning.")}</tbody></table></div>
    <h3>Large quakes in the next 20 days</h3>
    <div class="table-wrap"><table><thead><tr><th>When</th><th>Magnitude</th><th>Place</th></tr></thead><tbody>${quakes || emptyRow(3, "None.")}</tbody></table></div>
  `;
}

async function loadDay(targetId = "day-out") {
  const day = $("day-input").value;
  const el = $(targetId);
  if (!day || !el) return;
  el.innerHTML = `<p class="hint">Loading…</p>`;
  try {
    const data = await get(`/api/day?date=${day}`);
    el.innerHTML = dayHtml(data);
  } catch (err) {
    el.innerHTML = `<p class="empty">Could not load that morning: ${err.message}</p>`;
  }
}

function closeDrawer() {
  const drawer = $("day-drawer");
  if (drawer) drawer.hidden = true;
  document.body.classList.remove("drawer-open");
}

async function openDrawer(day) {
  const drawer = $("day-drawer");
  if (!drawer) return;
  drawer.hidden = false;
  document.body.classList.add("drawer-open");
  $("drawer-title").textContent = `Morning of ${day}`;
  await loadDay("drawer-body");
  $("drawer-close")?.focus();
}

function openMorning(day, { drawer = false } = {}) {
  if (!day) return;
  state.selectedDay = day;
  if ($("day-input")) $("day-input").value = day;
  if (drawer) {
    showTab("ledger");
    document.querySelectorAll(".ledger-row").forEach((row) => {
      row.classList.toggle("is-on", row.dataset.day === day);
    });
    openDrawer(day);
    return;
  }
  closeDrawer();
  showTab("overview");
  loadDay();
  $("day-out")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function lineChart(id, labels, datasets) {
  if (state.charts[id]) state.charts[id].destroy();
  const ctx = $(id);
  if (!ctx) return;
  state.charts[id] = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { labels: { color: "#8b96a8", boxWidth: 10 } } },
      scales: {
        x: { ticks: { color: "#8b96a8", maxTicksLimit: 8 }, grid: { color: "#243041" } },
        y: { ticks: { color: "#8b96a8" }, grid: { color: "#243041" } },
      },
    },
  });
  ctx.parentElement.style.height = "220px";
}

async function loadSeries() {
  const start = $("ser-start").value;
  const end = $("ser-end").value;
  let data;
  try {
    data = await get(`/api/series?start=${start}&end=${end}`);
  } catch (err) {
    console.warn(err);
    return;
  }
  const days = data.days || [];
  const labels = days.map((d) => (d.issue_time || "").slice(0, 10));
  lineChart("c-spf", labels, [{
    label: "solar proton window",
    data: days.map((d) => d.spf_significant_window),
    borderColor: "#e8a04a",
    backgroundColor: "rgba(232,160,74,0.22)",
    fill: true,
    pointRadius: 0,
    stepped: true,
  }]);
  lineChart("c-proton", labels, [{
    label: "proton density",
    data: days.map((d) => d.proton_density_cm3),
    borderColor: "#3d9a7a",
    pointRadius: 0,
    borderWidth: 1.4,
  }, {
    label: "spike",
    data: days.map((d) => (d.proton_spike ? d.proton_density_cm3 : null)),
    borderColor: "#f0c38a",
    pointRadius: 2,
    showLine: false,
  }]);
  lineChart("c-storm", labels, [{
    label: "Kp",
    data: days.map((d) => d.kp),
    borderColor: "#e8edf4",
    pointRadius: 0,
    borderWidth: 1.2,
  }, {
    label: "Dst",
    data: days.map((d) => d.dst_nt),
    borderColor: "#d15b4a",
    pointRadius: 0,
    borderWidth: 1.2,
  }]);
  lineChart("c-cat", labels, [{
    label: "M≥6 in prior 7d",
    data: days.map((d) => d.n_m6_7d),
    borderColor: "#e8a04a",
    pointRadius: 0,
    borderWidth: 1.2,
  }]);
  if ($("c-tec") && days.some((d) => d.tec_tecu != null || d.tec_nz_tecu != null)) {
    lineChart("c-tec", labels, [{
      label: "Earth-wide",
      data: days.map((d) => d.tec_tecu),
      borderColor: "#e8edf4",
      pointRadius: 0,
      borderWidth: 1.2,
    }, {
      label: "New Zealand",
      data: days.map((d) => d.tec_nz_tecu),
      borderColor: "#3d9a7a",
      pointRadius: 0,
      borderWidth: 1.2,
    }]);
  }
  if ($("c-tec-boxes") && days.some((d) => d.tec_hikurangi_tecu != null || d.tec_japan_tecu != null)) {
    lineChart("c-tec-boxes", labels, [{
      label: "Hikurangi",
      data: days.map((d) => d.tec_hikurangi_tecu),
      borderColor: "#e8a04a",
      pointRadius: 0,
      borderWidth: 1.2,
    }, {
      label: "Alpine Fault",
      data: days.map((d) => d.tec_alpine_tecu),
      borderColor: "#3d9a7a",
      pointRadius: 0,
      borderWidth: 1.2,
    }, {
      label: "Kermadec",
      data: days.map((d) => d.tec_kermadec_tecu),
      borderColor: "#d15b4a",
      pointRadius: 0,
      borderWidth: 1.2,
    }, {
      label: "Japan (practice)",
      data: days.map((d) => d.tec_japan_tecu),
      borderColor: "#5b8def",
      pointRadius: 0,
      borderWidth: 1.2,
    }, {
      label: "Chile (practice)",
      data: days.map((d) => d.tec_chile_tecu),
      borderColor: "#c49ad4",
      pointRadius: 0,
      borderWidth: 1.2,
    }]);
  }
  if (days.some((d) => d.coulomb_hikurangi != null || d.coulomb_alpine != null)) {
    lineChart("c-coulomb", labels, [{
      label: "Hikurangi",
      data: days.map((d) => d.coulomb_hikurangi),
      borderColor: "#e8a04a",
      pointRadius: 0,
      borderWidth: 1.2,
    }, {
      label: "Alpine Fault",
      data: days.map((d) => d.coulomb_alpine),
      borderColor: "#3d9a7a",
      pointRadius: 0,
      borderWidth: 1.2,
    }, {
      label: "Kermadec",
      data: days.map((d) => d.coulomb_kermadec),
      borderColor: "#d15b4a",
      pointRadius: 0,
      borderWidth: 1.2,
    }]);
  }
  if (days.some((d) => d.cusp_align_hikurangi != null || d.cusp_align_kermadec != null)) {
    lineChart("c-cusp", labels, [{
      label: "Hikurangi",
      data: days.map((d) => d.cusp_align_hikurangi),
      borderColor: "#e8a04a",
      pointRadius: 0,
      borderWidth: 1.2,
    }, {
      label: "Alpine Fault",
      data: days.map((d) => d.cusp_align_alpine),
      borderColor: "#3d9a7a",
      pointRadius: 0,
      borderWidth: 1.2,
    }, {
      label: "Kermadec",
      data: days.map((d) => d.cusp_align_kermadec),
      borderColor: "#d15b4a",
      pointRadius: 0,
      borderWidth: 1.2,
    }]);
  }
  if (days.some((d) => d.mig_along_hikurangi != null || d.mig_along_kermadec != null)) {
    lineChart("c-mig-along", labels, [{
      label: "Hikurangi",
      data: days.map((d) => d.mig_along_hikurangi),
      borderColor: "#e8a04a",
      pointRadius: 0,
      borderWidth: 1.2,
    }, {
      label: "Alpine Fault",
      data: days.map((d) => d.mig_along_alpine),
      borderColor: "#3d9a7a",
      pointRadius: 0,
      borderWidth: 1.2,
    }, {
      label: "Kermadec",
      data: days.map((d) => d.mig_along_kermadec),
      borderColor: "#d15b4a",
      pointRadius: 0,
      borderWidth: 1.2,
    }]);
    lineChart("c-mig-updip", labels, [{
      label: "Hikurangi",
      data: days.map((d) => d.mig_updip_hikurangi),
      borderColor: "#e8a04a",
      pointRadius: 0,
      borderWidth: 1.2,
    }, {
      label: "Alpine Fault",
      data: days.map((d) => d.mig_updip_alpine),
      borderColor: "#3d9a7a",
      pointRadius: 0,
      borderWidth: 1.2,
    }, {
      label: "Kermadec",
      data: days.map((d) => d.mig_updip_kermadec),
      borderColor: "#d15b4a",
      pointRadius: 0,
      borderWidth: 1.2,
    }]);
  }
}

function wireUi() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => showTab(btn.dataset.tab));
  });
  $("day-go")?.addEventListener("click", () => loadDay());
  const applyLedger = () => { state.offset = 0; loadLedger(); };
  $("ledger-go")?.addEventListener("click", applyLedger);
  ["f-model", "f-region", "f-kind", "f-hit"].forEach((id) => {
    $(id)?.addEventListener("change", applyLedger);
  });
  $("prev")?.addEventListener("click", () => {
    state.offset = Math.max(0, state.offset - 120);
    loadLedger();
  });
  $("next")?.addEventListener("click", () => {
    state.offset += 120;
    loadLedger();
  });
  $("ser-go")?.addEventListener("click", loadSeries);
  $("bt-go")?.addEventListener("click", runBacktest);
  $("brain-go")?.addEventListener("click", retrainBrain);
  $("globe-nz")?.addEventListener("click", lookAtNZ);
  $("globe-earth")?.addEventListener("click", lookAtEarth);
  $("globe-spin")?.addEventListener("click", () => setGlobeSpin(!state.globeSpin));
  document.querySelectorAll("[data-leftover-mode]").forEach((btn) => {
    btn.addEventListener("click", () => setLeftoverMode(btn.dataset.leftoverMode));
  });
  $("drawer-close")?.addEventListener("click", closeDrawer);
  document.querySelectorAll("[data-drawer-close]").forEach((el) => {
    el.addEventListener("click", closeDrawer);
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") closeDrawer();
  });
  window.addEventListener("resize", resizeGlobe);
  window.addEventListener("hashchange", () => {
    const { tab, source } = parseHash();
    if (tab !== state.tab) showTab(tab, source);
  });
}

async function boot() {
  try {
    const stored = localStorage.getItem(MODE_STORE);
    if (stored === "lastweek" || stored === "all") state.leftoverMode = stored;
  } catch (_) {
    /* ignore */
  }
  wireUi();
  syncModeButtons();
  const incoming = parseHash();
  const bench = fillBacktestDrivers().catch((err) => {
    if ($("bt-out")) $("bt-out").innerHTML = `<p class="empty">Bench failed to load: ${err.message}</p>`;
  });
  const mind = fillBrain().catch(() => {});
  const globe = get("/api/globe").then(renderWeek).catch((err) => {
    if ($("globe-meta")) $("globe-meta").textContent = `Could not load this week’s calls: ${err.message}`;
  });
  try {
    state.snap = await get("/api/snapshot", 3);
    await loadFreshHunt(state.snap);
    await Promise.all([bench, mind, globe]);
    renderOverview(state.snap);
    renderScoreboard(state.snap);
    applyLedgerFilters({ ...storedFilters(), ...incoming.filters });
    showTab(incoming.tab, incoming.source);
    await loadLedger();
    await loadDay();
  } catch (err) {
    $("lede").textContent = `The lab failed to load: ${err.message}.`;
    await Promise.all([bench, mind, globe]);
    showTab(incoming.tab, incoming.source);
  }
}

boot();
