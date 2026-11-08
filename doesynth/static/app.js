"use strict";

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const fmt = (x, d = 3) =>
  x == null || Number.isNaN(x) ? "—" : x.toLocaleString("ru-RU", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtTime = (s) => (s < 1 ? `${fmt(s * 1000, 0)} мс` : `${fmt(s, 2)} с`);
const png = (b64) => `data:image/png;base64,${b64}`;

const state = {
  config: null,
  images: [null, null, null, null], // base64 PNG 8 бит 128×128
  names: [null, null, null, null],
  mode: "net",
  last: null, // {id, results}
  batchId: null,
  busy: false,
  slotTarget: 0,
};

/* ---------- строка состояния ---------- */
function status(text, kind = "") {
  $("#status-text").textContent = text;
  $("#status").className = `status ${kind}`;
}
function busy(on, text) {
  state.busy = on;
  $("#progress").hidden = !on;
  for (const id of ["#btn-run", "#btn-run2", "#btn-batch-dir", "#btn-batch-zip"]) $(id).disabled = on;
  if (on) status(text || "Идёт расчёт…", "busy");
}

async function api(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch (_) { /* не JSON */ }
    throw new Error(msg);
  }
  return res;
}

/* ---------- входные изображения ---------- */
function renderSlots() {
  const z = state.config ? state.config.physics.distances_mm : [24.1, 25.3, 26.5, 27.7];
  const box = $("#slots");
  box.innerHTML = "";
  state.images.forEach((img, i) => {
    const slot = document.createElement("div");
    slot.className = `slot${img ? " filled" : ""}`;
    slot.title = img ? `${state.names[i]} — заменить` : "Выбрать изображение";
    const label = `${i + 1} · z = ${fmt(z[i], 1)} мм`;
    slot.innerHTML = img
      ? `<img src="${png(img)}" alt=""><div class="slot-label">${label}</div>`
      : `<div class="slot-empty"><b>+</b><span>Плоскость ${i + 1}</span><span>z = ${fmt(z[i], 1)} мм</span></div>`;
    slot.addEventListener("click", () => {
      state.slotTarget = i;
      $("#file-one").click();
    });
    slot.addEventListener("dragover", (e) => { e.preventDefault(); slot.classList.add("drag"); });
    slot.addEventListener("dragleave", () => slot.classList.remove("drag"));
    slot.addEventListener("drop", (e) => {
      e.preventDefault();
      slot.classList.remove("drag");
      const files = [...e.dataTransfer.files];
      if (files.length) loadFiles(files, i);
    });
    box.appendChild(slot);
  });
}

function showWarnings(list) {
  const w = $("#warnings");
  w.hidden = !list.length;
  w.innerHTML = list.map((t) => `<div>${escapeHtml(t)}</div>`).join("");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

async function loadFiles(files, start = null) {
  if (!files.length) return;
  if (start == null) {
    const free = state.images.findIndex((x) => !x);
    start = files.length === 1 && free >= 0 ? free : 0;
  }
  const form = new FormData();
  files.forEach((f) => form.append("files", f, f.name));
  try {
    status("Чтение изображений…", "busy");
    const res = await (await api("/api/images", { method: "POST", body: form })).json();
    const list = res.images;
    const warnings = [...res.warnings];
    if (files.length > 1) start = 0;
    if (list.length > 4) warnings.push(`Выбрано ${list.length} файлов — взяты первые 4 по имени.`);
    list.slice(0, 4).forEach((item, k) => {
      const i = files.length > 1 ? k : start;
      state.images[i] = item.image;
      state.names[i] = item.name;
    });
    showWarnings(warnings);
    renderSlots();
    readyStatus();
  } catch (e) {
    status(`Ошибка чтения: ${e.message}`, "error");
  }
}

async function loadExample(name) {
  try {
    status(`Загрузка примера «${name}»…`, "busy");
    const res = await (await api(`/api/examples/${encodeURIComponent(name)}`)).json();
    res.images.forEach((b64, i) => { state.images[i] = b64; state.names[i] = res.names[i]; });
    showWarnings([]);
    renderSlots();
    readyStatus(`Пример «${name}» загружен`);
  } catch (e) {
    status(`Ошибка: ${e.message}`, "error");
  }
}

function readyStatus(prefix) {
  const n = state.images.filter(Boolean).length;
  const tail = n === 4 ? "можно синтезировать" : `загружено ${n} из 4 изображений`;
  status(prefix ? `${prefix} — ${tail}` : `Готово — ${tail}`);
}

/* ---------- параметры метода ---------- */
function benchLine(b) {
  if (!b) return "";
  const parts = [];
  if (b.finetune)
    parts.push(`с дооптимизацией (${b.finetune.steps} шагов) — SSIM ${fmt(b.finetune.ssim)}, Пирсон ${fmt(b.finetune.pearson)}`);
  if (b.net)
    parts.push(`без неё — SSIM ${fmt(b.net.ssim)}, Пирсон ${fmt(b.net.pearson)}` + (b.net_s ? `, ${fmtTime(b.net_s)}` : ""));
  if (b.ssim != null)
    parts.push(`${b.iters} итераций — SSIM ${fmt(b.ssim)}, Пирсон ${fmt(b.pearson)}` + (b.seconds ? `, ${fmtTime(b.seconds)}` : ""));
  return parts.join("; ");
}

function updateNetNote() {
  const id = $("#net-model").value;
  const m = state.config.nets.find((x) => x.id === id);
  if (!m) return;
  const bench = state.config.bench;
  const line = benchLine(m.bench);
  $("#net-note").innerHTML =
    (m.default ? `<div class="best">★ Лучшая по метрикам — выбрана по умолчанию</div>` : "") +
    `<div>${escapeHtml(m.description)}</div>` +
    (line ? `<div class="bench">${bench ? escapeHtml(bench.label) + ": " : ""}${line}.</div>` : "");
}

function updateIterNote() {
  const id = $("#iter-method").value;
  const m = state.config.iterative.find((x) => x.id === id);
  if (!m) return;
  const bench = state.config.bench;
  const line = benchLine(m.bench);
  $("#iter-note").innerHTML =
    `<div>${escapeHtml(m.description)}</div>` +
    (line ? `<div class="bench">${bench ? escapeHtml(bench.label) + ": " : ""}${line}.</div>` : "");
  $("#iter-unit").textContent = m.unit === "шагов" ? "Шагов" : "Итераций";
  $("#iter-n").max = m.max_iters;
  $("#iter-n").value = m.default_iters;
}

function setMode(mode) {
  state.mode = mode;
  $$("#mode button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  $("#group-net").hidden = mode === "iter";
  $("#group-iter").hidden = mode === "net";
}

function settings() {
  return {
    mode: state.mode,
    net: {
      model: $("#net-model").value,
      finetune: $("#net-finetune").checked,
      steps: Math.max(1, parseInt($("#net-steps").value, 10) || 1),
    },
    iter: {
      method: $("#iter-method").value,
      iters: Math.max(1, parseInt($("#iter-n").value, 10) || 1),
    },
  };
}

/* ---------- синтез ---------- */
async function run() {
  if (state.busy) return;
  if (state.images.some((x) => !x)) {
    status("Нужно 4 изображения — по одному на плоскость", "error");
    return;
  }
  const s = settings();
  busy(true, s.mode === "compare" ? "Синтез: нейросеть и итерационный метод…" : "Синтез…");
  try {
    const res = await (
      await api("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images: state.images, ...s }),
      })
    ).json();
    state.last = res;
    renderResults(res);
    $("#btn-save-doe").disabled = false;
    $("#btn-save-all").disabled = false;
    const r0 = res.results[0];
    status(`Готово — ${r0.label}: SSIM ${fmt(r0.metrics.ssim_mean)}, Пирсон ${fmt(r0.metrics.pearson_mean)}, синтез ${fmtTime(r0.seconds)}`);
  } catch (e) {
    status(`Ошибка синтеза: ${e.message}`, "error");
  } finally {
    busy(false);
    $("#progress").hidden = true;
  }
}

function renderResults(res) {
  $("#empty").hidden = true;
  const box = $("#results");
  box.innerHTML = "";
  if (res.results.length > 1) box.appendChild(compareTable(res.results));
  res.results.forEach((r, i) => box.appendChild(resultCard(r, i, res.id)));
}

function compareTable(results) {
  const [a, b] = results;
  const panel = document.createElement("section");
  panel.className = "panel";
  const best = (x, y, higher = true) => (higher ? x > y : x < y);
  const row = (r, other, name) => `
    <tr><td>${name}<div class="sub">${escapeHtml(r.label)}</div></td>
      <td class="${best(r.metrics.ssim_mean, other.metrics.ssim_mean) ? "win" : ""}">${fmt(r.metrics.ssim_mean)}</td>
      <td class="${best(r.metrics.pearson_mean, other.metrics.pearson_mean) ? "win" : ""}">${fmt(r.metrics.pearson_mean)}</td>
      <td class="${best(r.seconds, other.seconds, false) ? "win" : ""}">${fmtTime(r.seconds)}</td>
      <td>${fmtTime(r.total_seconds)}</td></tr>`;
  const speed = b.seconds / Math.max(a.seconds, 1e-6);
  const dS = a.metrics.ssim_mean - b.metrics.ssim_mean;
  const dP = a.metrics.pearson_mean - b.metrics.pearson_mean;
  const sign = (x) => (x >= 0 ? "+" : "−") + fmt(Math.abs(x));
  const speedText =
    speed >= 1.1 ? `быстрее в ${fmt(speed, 1)} раза` :
    speed <= 0.9 ? `медленнее в ${fmt(1 / speed, 1)} раза` : "по времени наравне с итерационным методом";
  panel.innerHTML = `
    <div class="panel-head"><div class="panel-title">Сравнение методов</div></div>
    <div class="panel-body">
      <table class="compare-table">
        <tr><th>Метод</th><th>SSIM</th><th>Пирсон</th><th>Время синтеза</th><th>Полное время</th></tr>
        ${row(a, b, "Нейросеть")}${row(b, a, "Итерационный")}
      </table>
      <div class="verdict">Нейросеть ${speedText}; разница качества: SSIM ${sign(dS)}, Пирсон ${sign(dP)}.</div>
    </div>`;
  return panel;
}

function resultCard(r, index, rid) {
  const node = $("#tpl-result").content.firstElementChild.cloneNode(true);
  const f = (name) => $(`[data-f="${name}"]`, node);
  f("label").textContent = r.label;
  f("ssim").textContent = fmt(r.metrics.ssim_mean);
  f("pearson").textContent = fmt(r.metrics.pearson_mean);
  f("time").textContent = fmtTime(r.seconds);
  f("time").title = `Полное время с восстановлением и метриками: ${fmtTime(r.total_seconds)}`;
  f("doe").src = png(r.doe);
  const link = f("doelink");
  link.href = png(r.doe);
  link.download = r.kind === "iter" ? "doe_iterative.png" : "doe.png";

  const z = state.config.physics.distances_mm;
  const planes = f("planes");
  planes.innerHTML = `<div></div>` + z.map((zz, i) => `<div class="plane-head">Плоскость <b>${i + 1}</b> · z = ${fmt(zz, 1)} мм</div>`).join("");
  planes.insertAdjacentHTML("beforeend", `<div class="row-label">Исходное</div>`);
  state.images.forEach((img) => planes.insertAdjacentHTML("beforeend", `<div class="plane"><img src="${png(img)}" alt=""></div>`));
  planes.insertAdjacentHTML("beforeend", `<div class="row-label">Восстановление</div>`);
  r.recons.forEach((img, i) =>
    planes.insertAdjacentHTML(
      "beforeend",
      `<div class="plane"><img src="${png(img)}" alt="">
        <div class="plane-metrics">SSIM <b>${fmt(r.metrics.ssim[i])}</b> · Пирсон <b>${fmt(r.metrics.pearson[i])}</b></div></div>`
    )
  );

  // просмотр по глубине
  const range = f("z");
  const zimg = f("zimg");
  const zval = f("zval");
  let timer = null;
  const update = () => {
    const zz = parseFloat(range.value);
    zval.textContent = `${fmt(zz, 1)} мм`;
    clearTimeout(timer);
    timer = setTimeout(() => { zimg.src = `/api/slice/${rid}/${index}?z=${zz}`; }, 60);
  };
  range.value = z[0];
  range.addEventListener("input", update);
  update();
  return node;
}

/* ---------- сохранение ---------- */
function saveDoe() {
  if (!state.last) return;
  const r = state.last.results[0];
  const a = document.createElement("a");
  a.href = png(r.doe);
  a.download = "doe.png";
  a.click();
}
function saveAll(id) {
  if (!id) return;
  window.location.href = `/api/download/${id}`;
}

/* ---------- пакет ---------- */
async function runBatch(fileList, isZip) {
  const files = [...fileList];
  if (!files.length || state.busy) return;
  const form = new FormData();
  const paths = [];
  files.forEach((f) => {
    form.append("files", f, f.name);
    paths.push(isZip ? f.name : f.webkitRelativePath || f.name);
  });
  form.append("paths", JSON.stringify(paths));
  form.append("settings", JSON.stringify(settings()));
  busy(true, "Пакетная обработка…");
  try {
    const res = await (await api("/api/batch", { method: "POST", body: form })).json();
    state.batchId = res.id;
    $("#btn-batch-save").disabled = !res.id;
    renderBatch(res);
    const n = res.kits.filter((k) => !k.error).length;
    status(`Готово — обработано комплектов: ${n} из ${res.kits.length}`);
  } catch (e) {
    status(`Ошибка пакетной обработки: ${e.message}`, "error");
  } finally {
    busy(false);
    $("#progress").hidden = true;
  }
}

function renderBatch(res) {
  const kinds = Object.keys(res.means);
  const title = { net: "Нейросеть", iter: "Итерационный" };
  const head =
    `<tr><th>Комплект</th><th>ДОЭ</th>` +
    kinds.map((k) => `<th>SSIM${kinds.length > 1 ? ` · ${title[k]}` : ""}</th><th>Пирсон</th><th>Синтез</th>`).join("") +
    `</tr>`;
  const rows = res.kits
    .map((k) => {
      if (k.error) return `<tr><td>${escapeHtml(k.kit)}</td><td class="err" colspan="${1 + 3 * kinds.length}">${escapeHtml(k.error)}</td></tr>`;
      const cells = kinds
        .map((kind) => {
          const r = k.results.find((x) => x.kind === kind);
          return `<td>${fmt(r.ssim_mean)}</td><td>${fmt(r.pearson_mean)}</td><td>${fmtTime(r.seconds)}</td>`;
        })
        .join("");
      return `<tr><td>${escapeHtml(k.kit)}<div class="sub">${k.files.map(escapeHtml).join(", ")}</div></td><td><img src="${png(k.doe)}" alt=""></td>${cells}</tr>`;
    })
    .join("");
  const mean =
    `<tr class="mean"><td>Среднее</td><td></td>` +
    kinds.map((k) => `<td>${fmt(res.means[k].ssim_mean)}</td><td>${fmt(res.means[k].pearson_mean)}</td><td>${fmtTime(res.means[k].seconds)}</td>`).join("") +
    `</tr>`;
  const methods = kinds.map((k) => `${title[k]}: ${escapeHtml(res.means[k].method)}`).join("; ");
  $("#batch-out").innerHTML = `<div class="hint">${methods}</div><table class="batch-table">${head}${rows}${kinds.length ? mean : ""}</table>`;
}

/* ---------- запуск ---------- */
// start.bat и start.sh открывают интерфейс отдельным окном (режим приложения Edge или Chrome)
// с ?window=app. При первом открытии окна ставим его по центру экрана, а если браузер не
// применил --window-size и окно маленькое, увеличиваем до рабочего размера. При перезагрузке
// страницы окно не трогаем: пользователь мог его передвинуть.
function fitAppWindow() {
  if (new URLSearchParams(window.location.search).get("window") !== "app") return;
  try {
    if (sessionStorage.getItem("doe-window-fitted")) return;
    sessionStorage.setItem("doe-window-fitted", "1");
  } catch (_) {
    /* без sessionStorage просто подгоняем окно при каждой загрузке */
  }
  const s = window.screen;
  let w = window.outerWidth;
  let h = window.outerHeight;
  try {
    if (w < Math.min(1440, s.availWidth) - 16 || h < Math.min(900, s.availHeight) - 16) {
      w = Math.min(1440, s.availWidth);
      h = Math.min(900, s.availHeight);
      window.resizeTo(w, h);
    }
    window.moveTo((s.availLeft || 0) + Math.max(0, Math.round((s.availWidth - w) / 2)),
      (s.availTop || 0) + Math.max(0, Math.round((s.availHeight - h) / 2)));
  } catch (_) {
    /* браузер не дал изменить окно — остаётся как есть */
  }
}

async function init() {
  fitAppWindow();
  renderSlots();
  try {
    state.config = await (await api("/api/config")).json();
  } catch (e) {
    status(`Сервер недоступен: ${e.message}`, "error");
    return;
  }
  const cfg = state.config;
  $("#device").textContent = `v${cfg.version} · ${cfg.device}`;
  const p = cfg.physics;
  $("#physics").textContent = `λ = ${fmt(p.wavelength_nm, 0)} нм · Δx = ${fmt(p.pixel_um, 0)} мкм · ${p.n}×${p.n} · z = ${p.distances_mm.map((z) => fmt(z, 1)).join(" / ")} мм`;

  const sel = $("#net-model");
  const groups = {};
  cfg.nets.forEach((m) => {
    if (!groups[m.group]) {
      groups[m.group] = document.createElement("optgroup");
      groups[m.group].label = m.group;
      sel.appendChild(groups[m.group]);
    }
    const o = new Option(m.default ? `★ ${m.title}` : m.title, m.id);
    groups[m.group].appendChild(o);
  });
  sel.value = cfg.defaults.model;
  sel.addEventListener("change", updateNetNote);
  $("#net-finetune").checked = cfg.defaults.finetune;
  $("#net-steps").value = cfg.defaults.finetune_steps;
  $("#net-steps").max = cfg.defaults.max_finetune_steps;
  const syncSteps = () => { $("#net-steps").disabled = !$("#net-finetune").checked; };
  $("#net-finetune").addEventListener("change", syncSteps);
  syncSteps();
  updateNetNote();

  const isel = $("#iter-method");
  cfg.iterative.forEach((m) => isel.appendChild(new Option(m.title, m.id)));
  isel.value = cfg.defaults.iter_method;
  isel.addEventListener("change", updateIterNote);
  updateIterNote();

  const menu = $("#menu-examples");
  if (!cfg.examples.length) menu.innerHTML = `<button disabled>Примеров нет</button>`;
  cfg.examples.forEach((name) => {
    const b = document.createElement("button");
    b.textContent = name;
    b.addEventListener("click", () => { menu.hidden = true; loadExample(name); });
    menu.appendChild(b);
  });
  renderSlots();
  readyStatus();
  // ?demo=1 — загрузить первый пример и сразу синтезировать (для демонстрации и скриншотов)
  const params = new URLSearchParams(window.location.search);
  if (params.get("demo") && cfg.examples.length) {
    if (params.get("mode")) setMode(params.get("mode"));
    await loadExample(params.get("example") || cfg.examples[0]);
    await run();
  }
}

function bind() {
  $("#btn-open").addEventListener("click", () => $("#file-four").click());
  $("#btn-open2").addEventListener("click", () => $("#file-four").click());
  $("#file-four").addEventListener("change", (e) => { loadFiles([...e.target.files]); e.target.value = ""; });
  $("#file-one").addEventListener("change", (e) => { loadFiles([...e.target.files].slice(0, 1), state.slotTarget); e.target.value = ""; });
  $("#btn-clear").addEventListener("click", () => {
    state.images = [null, null, null, null];
    state.names = [null, null, null, null];
    showWarnings([]);
    renderSlots();
    readyStatus();
  });
  $("#btn-examples").addEventListener("click", (e) => { e.stopPropagation(); $("#menu-examples").hidden = !$("#menu-examples").hidden; });
  document.addEventListener("click", () => { $("#menu-examples").hidden = true; });
  $("#btn-run").addEventListener("click", run);
  $("#btn-run2").addEventListener("click", run);
  $("#btn-save-doe").addEventListener("click", saveDoe);
  $("#btn-save-all").addEventListener("click", () => saveAll(state.last && state.last.id));
  $$("#mode button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $$(".tab").forEach((t) =>
    t.addEventListener("click", () => {
      $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
      $("#view-single").hidden = t.dataset.tab !== "single";
      $("#view-batch").hidden = t.dataset.tab !== "batch";
    })
  );
  $("#btn-batch-dir").addEventListener("click", () => $("#file-dir").click());
  $("#btn-batch-zip").addEventListener("click", () => $("#file-zip").click());
  $("#file-dir").addEventListener("change", (e) => { runBatch(e.target.files, false); e.target.value = ""; });
  $("#file-zip").addEventListener("change", (e) => { runBatch(e.target.files, true); e.target.value = ""; });
  $("#btn-batch-save").addEventListener("click", () => saveAll(state.batchId));
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); run(); }
  });
  // перетаскивание четырёх файлов в любое место окна
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => {
    if (e.target.closest(".slot")) return;
    e.preventDefault();
    const files = [...e.dataTransfer.files];
    if (files.length) loadFiles(files);
  });
}

bind();
init();
