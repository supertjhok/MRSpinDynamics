const state = {
  options: null,
  results: [],
  selectedId: null,
  selected: null,
  siteColors: new Map()
};

const els = {
  stats: document.getElementById("stats"),
  homeStats: document.getElementById("homeStats"),
  searchForm: document.getElementById("searchForm"),
  query: document.getElementById("query"),
  homeButton: document.getElementById("homeButton"),
  category: document.getElementById("category"),
  isotope: document.getElementById("isotope"),
  sourceType: document.getElementById("sourceType"),
  freqMin: document.getElementById("freqMin"),
  freqMax: document.getElementById("freqMax"),
  clearFilters: document.getElementById("clearFilters"),
  resultCount: document.getElementById("resultCount"),
  results: document.getElementById("results"),
  emptyState: document.getElementById("emptyState"),
  detail: document.getElementById("detail"),
  compoundName: document.getElementById("compoundName"),
  compoundMeta: document.getElementById("compoundMeta"),
  pubchemLink: document.getElementById("pubchemLink"),
  structureBox: document.getElementById("structureBox"),
  structureState: document.getElementById("structureState"),
  spectrumPlot: document.getElementById("spectrumPlot"),
  spectrumRange: document.getElementById("spectrumRange"),
  measurementCount: document.getElementById("measurementCount"),
  measurements: document.getElementById("measurements"),
  references: document.getElementById("references"),
  sources: document.getElementById("sources")
};

init();

async function init() {
  els.searchForm.addEventListener("submit", event => {
    event.preventDefault();
    search();
  });
  els.homeButton.addEventListener("click", showHome);
  [els.category, els.isotope, els.sourceType, els.freqMin, els.freqMax].forEach(el => {
    el.addEventListener("change", search);
  });
  els.query.addEventListener("input", debounce(search, 220));
  els.clearFilters.addEventListener("click", () => {
    els.query.value = "";
    els.category.value = "";
    els.isotope.value = "";
    els.sourceType.value = "";
    els.freqMin.value = "";
    els.freqMax.value = "";
    search();
  });
  const [stats, options] = await Promise.all([
    getJson("/api/stats"),
    getJson("/api/options")
  ]);
  renderStats(stats);
  renderOptions(options);
  await search();
}

function renderStats(stats) {
  const counts = stats.counts || {};
  els.stats.textContent = [
    `${counts.compounds || 0} compounds`,
    `${counts.sites || 0} sites`,
    `${counts.lines || 0} lines`,
    `${counts.references || 0} references`,
    `${counts.sources || 0} sources`
  ].join("  ");
  if (els.homeStats) {
    els.homeStats.innerHTML = [
      statTile(counts.compounds || 0, "compounds"),
      statTile(counts.sites || 0, "sites"),
      statTile(counts.lines || 0, "frequency lines"),
      statTile(counts.references || 0, "references")
    ].join("");
  }
}

function statTile(value, label) {
  return `<div class="stat-tile"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`;
}

function renderOptions(options) {
  state.options = options;
  fillSelect(els.category, "All categories", options.categories || []);
  fillSelect(els.isotope, "All isotopes", options.isotopes || []);
  fillSelect(els.sourceType, "All sources", options.source_types || []);
}

function fillSelect(select, firstLabel, values) {
  select.innerHTML = "";
  const first = document.createElement("option");
  first.value = "";
  first.textContent = firstLabel;
  select.appendChild(first);
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = humanSourceType(value);
    select.appendChild(option);
  }
}

async function search() {
  const params = new URLSearchParams();
  if (els.query.value.trim()) params.set("q", els.query.value.trim());
  if (els.category.value) params.set("category", els.category.value);
  if (els.isotope.value) params.set("isotope", els.isotope.value);
  if (els.sourceType.value) params.set("source_type", els.sourceType.value);
  if (els.freqMin.value) params.set("freq_min", els.freqMin.value);
  if (els.freqMax.value) params.set("freq_max", els.freqMax.value);
  const data = await getJson(`/api/search?${params.toString()}`);
  state.results = data.rows || [];
  renderResults();
  if (!state.results.length) {
    clearDetail();
  } else if (state.selectedId && !state.results.some(row => row.id === state.selectedId)) {
    showHome();
  } else {
    markSelected();
  }
}

function renderResults() {
  els.resultCount.textContent = `${state.results.length} shown`;
  els.results.innerHTML = "";
  for (const row of state.results) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "result-item";
    button.dataset.id = row.id;
    button.innerHTML = `
      <div class="result-title">${escapeHtml(row.canonical_name)}</div>
      <div class="formula">${formatFormula(row.conventional_formula || row.formula || "")}</div>
      <div class="chip-row">
        ${chip(row.category || "uncategorized", "strong")}
        ${chip(`${row.line_count || 0} lines`)}
        ${row.min_frequency_khz !== null ? chip(`${formatNumber(row.min_frequency_khz)}-${formatNumber(row.max_frequency_khz)} kHz`) : ""}
      </div>
      <div class="chip-row">${(row.isotopes || []).map(value => chip(value)).join("")}</div>
    `;
    button.addEventListener("click", () => selectCompound(row.id));
    els.results.appendChild(button);
  }
  markSelected();
}

async function selectCompound(id) {
  state.selectedId = id;
  state.selected = await getJson(`/api/compound/${encodeURIComponent(id)}`);
  renderDetail();
  markSelected();
}

function markSelected() {
  document.querySelectorAll(".result-item").forEach(item => {
    item.classList.toggle("active", item.dataset.id === state.selectedId);
  });
}

function clearDetail() {
  state.selectedId = null;
  state.selected = null;
  state.siteColors = new Map();
  els.emptyState.classList.remove("hidden");
  els.detail.classList.add("hidden");
}

function showHome() {
  clearDetail();
  markSelected();
}

function renderDetail() {
  const compound = state.selected;
  if (!compound) {
    clearDetail();
    return;
  }
  els.emptyState.classList.add("hidden");
  els.detail.classList.remove("hidden");
  els.compoundName.textContent = compound.canonical_name;
  els.compoundMeta.innerHTML = [
    chip(compound.category || "uncategorized", "strong"),
    compound.conventional_formula || compound.formula ? `<span class="formula">${formatFormula(compound.conventional_formula || compound.formula)}</span>` : "",
    consistencySummaryChip(compound.consistency_summary),
    ...(compound.aliases || []).slice(0, 6).map(alias => chip(alias))
  ].join("");
  const searchUrl = compound.structure && compound.structure.pubchem_search_url;
  els.pubchemLink.href = searchUrl || "https://pubchem.ncbi.nlm.nih.gov/";
  els.pubchemLink.style.visibility = searchUrl ? "visible" : "hidden";
  assignSiteColors(compound.samples || []);
  renderStructure(compound);
  renderSpectrum(compound.spectrum || []);
  renderMeasurements(compound.samples || []);
  renderReferences(compound.references || []);
  renderSources(compound.sources || []);
}

function renderStructure(compound) {
  const structure = compound.structure || {};
  const candidates = structure.candidates || [];
  els.structureBox.innerHTML = "";
  if (!candidates.length) {
    renderFormulaFallback(structure.formula, "No structure lookup candidate");
    return;
  }
  els.structureState.textContent = `trying ${candidates[0].label}`;
  const img = document.createElement("img");
  let index = 0;
  img.alt = `Structure for ${compound.canonical_name}`;
  img.onload = () => {
    els.structureState.textContent = candidates[index].label;
  };
  img.onerror = () => {
    index += 1;
    if (index < candidates.length) {
      els.structureState.textContent = `trying ${candidates[index].label}`;
      img.src = candidates[index].image_url;
    } else {
      renderFormulaFallback(structure.formula, "No PubChem image found");
    }
  };
  img.src = candidates[index].image_url;
  els.structureBox.appendChild(img);
}

function renderFormulaFallback(formula, label) {
  els.structureState.textContent = label;
  els.structureBox.innerHTML = `
    <div class="formula-card">
      <div class="muted">Structure diagram unavailable</div>
      <div class="formula">${formatFormula(formula || "formula unavailable")}</div>
    </div>
  `;
}

function renderSpectrum(points) {
  els.spectrumPlot.innerHTML = "";
  if (!points.length) {
    els.spectrumRange.textContent = "no lines";
    els.spectrumPlot.innerHTML = `<div class="empty-state">No frequency lines recorded.</div>`;
    return;
  }
  const min = Math.min(...points.map(point => point.frequency_khz));
  const max = Math.max(...points.map(point => point.frequency_khz));
  const span = Math.max(max - min, 1);
  els.spectrumRange.textContent = `${formatNumber(min)}-${formatNumber(max)} kHz`;
  const visiblePoints = points.slice(0, 180).map(point => ({
    ...point,
    plotLeft: 4 + ((point.frequency_khz - min) / span) * 92
  }));
  for (const point of visiblePoints) {
    const line = document.createElement("div");
    line.className = "spectrum-line";
    line.style.left = `${point.plotLeft}%`;
    line.style.height = `${44 + Math.min(110, (point.frequency_khz - min) / span * 74)}px`;
    line.style.background = siteColor(point.site_id);
    line.title = [
      `${formatNumber(point.frequency_khz)} kHz`,
      point.isotope,
      point.site_label,
      point.sample_label,
      point.temperature_original || (point.temperature_k !== null ? `${formatNumber(point.temperature_k)} K` : ""),
      point.method_description || point.method,
      point.source_type && humanSourceType(point.source_type)
    ].filter(Boolean).join(" | ");
    els.spectrumPlot.appendChild(line);
  }
  renderSpectrumLabels(visiblePoints);
  renderSpectrumLegend(visiblePoints);
  addAxisLabel(`${formatNumber(min)} kHz`, "8px");
  addAxisLabel(`${formatNumber(max)} kHz`, "calc(100% - 84px)");
}

function renderSpectrumLabels(points) {
  const rows = [];
  const maxRows = 7;
  for (const point of [...points].sort((a, b) => a.plotLeft - b.plotLeft)) {
    let rowIndex = rows.findIndex(lastLeft => point.plotLeft - lastLeft >= 9);
    if (rowIndex === -1) {
      rowIndex = Math.min(rows.length, maxRows - 1);
    }
    rows[rowIndex] = point.plotLeft;
    const label = document.createElement("div");
    label.className = "spectrum-label";
    label.style.left = `${point.plotLeft}%`;
    label.style.top = `${12 + rowIndex * 18}px`;
    label.style.borderColor = siteColor(point.site_id);
    label.textContent = `${formatNumber(point.frequency_khz)}${point.temperature_original ? ` @ ${formatTemperature(point.temperature_original)}` : ""}`;
    label.title = point.title || `${formatNumber(point.frequency_khz)} kHz`;
    els.spectrumPlot.appendChild(label);
  }
}

function renderSpectrumLegend(points) {
  const sites = [];
  const seen = new Set();
  for (const point of points) {
    const key = point.site_id || "unassigned";
    if (seen.has(key)) continue;
    seen.add(key);
    sites.push(point);
  }
  if (sites.length <= 1) return;
  const legend = document.createElement("div");
  legend.className = "spectrum-legend";
  legend.innerHTML = sites.slice(0, 8).map(point => `
    <span><i style="background:${siteColor(point.site_id)}"></i>${escapeHtml(point.site_label || point.isotope || "site")}</span>
  `).join("");
  els.spectrumPlot.appendChild(legend);
}

function addAxisLabel(text, left) {
  const label = document.createElement("div");
  label.className = "axis-label";
  label.style.left = left;
  label.textContent = text;
  els.spectrumPlot.appendChild(label);
}

function renderMeasurements(samples) {
  const sampleCount = samples.length;
  const lineCount = samples.reduce((sum, sample) => (
    sum + sample.sites.reduce((siteSum, site) => siteSum + site.lines.length, 0)
  ), 0);
  els.measurementCount.textContent = `${sampleCount} samples, ${lineCount} lines`;
  els.measurements.innerHTML = "";
  for (const sample of samples) {
    const group = document.createElement("div");
    group.className = "measurement-group";
    group.innerHTML = `
      <div class="sample-title">
        <div>
          <span>${escapeHtml(sample.label)}</span>
          <div class="sample-meta">${measurementChips(sample.measurement, sample).join("")}</div>
        </div>
        <span class="muted">${sample.sites.length} sites</span>
      </div>
      <div class="site-list">
        ${sample.sites.map(site => siteBlock(sample, site)).join("")}
      </div>
    `;
    els.measurements.appendChild(group);
  }
}

function siteBlock(sample, site) {
  const color = siteColor(site.id);
  return `
    <section class="site-block" style="--site-color:${color}">
      <div class="site-header">
        <div>
          <div class="site-title">${escapeHtml(site.isotope || "unknown isotope")} ${escapeHtml(site.site_label || "site")}</div>
          <div class="site-kind">${escapeHtml(siteKind(site))}</div>
          <div class="site-meta">${measurementChips(site.measurement || {}, {}).join("")}</div>
        </div>
        <div class="site-params">${formatCoupling(site)}</div>
      </div>
      ${consistencyBanner(site.consistency)}
      ${site.lines.length ? lineTable(site.lines, sample, site) : `<div class="site-empty">No frequency line assigned to this site record.</div>`}
    </section>
  `;
}

function consistencySummaryChip(summary) {
  if (!summary || !summary.checked) return "";
  if (summary.flagged) {
    const label = summary.flagged === 1 ? "1 site flagged" : `${summary.flagged} sites flagged`;
    return `<span class="chip consistency-chip flagged" title="Simulator check: stored parameters disagree with measured lines">⚠ ${label}</span>`;
  }
  return `<span class="chip consistency-chip verified" title="Stored parameters reproduce the measured lines">✓ simulator-verified</span>`;
}

function consistencyBanner(flag) {
  if (!flag) return "";
  const flagged = Number(flag.flagged) === 1;
  const cls = flagged ? "flagged" : "verified";
  const icon = flagged ? "⚠" : "✓";
  const heading = flagged ? "Parameter / line inconsistency" : "Simulator-verified";
  const detail = flag.detail ? `<div class="consistency-detail">${escapeHtml(flag.detail)}</div>` : "";
  let implied = "";
  if (flagged && flag.implied_qcc_hz !== null && flag.implied_qcc_hz !== undefined) {
    implied = `<div class="consistency-implied">Lines imply C<sub>Q</sub> ${formatNumber(flag.implied_qcc_hz / 1e3)} kHz, η ${formatNumber(flag.implied_eta)}</div>`;
  }
  return `
    <div class="consistency-banner ${cls}">
      <span class="consistency-icon">${icon}</span>
      <div>
        <div class="consistency-heading">${heading}</div>
        ${detail}
        ${implied}
      </div>
    </div>
  `;
}

function lineTable(lines, sample, site) {
  return `
    <table class="line-table">
      <thead>
        <tr>
          <th>ν</th>
          <th>Condition</th>
          <th>Relaxation / width</th>
          <th>Source</th>
        </tr>
      </thead>
      <tbody>${lines.map(line => lineRow(sample, site, line)).join("")}</tbody>
    </table>
  `;
}

function lineRow(sample, site, line) {
  return `
    <tr>
      <td><strong>${formatNumber(line.frequency_khz)} kHz</strong><br><span class="muted">${escapeHtml(line.frequency_original || "")}</span></td>
      <td>${conditionDetails(line, sample)}</td>
      <td>${lineDetails(line)}</td>
      <td>${escapeHtml(humanSourceType(line.source_type || site.source_type || ""))}</td>
    </tr>
  `;
}

function formatCoupling(site) {
  const parts = [];
  if (site.qcc_khz !== null) parts.push(`<span><span class="symbol">C<sub>Q</sub></span> ${formatNumber(site.qcc_khz)} kHz</span>`);
  if (site.eta !== null) parts.push(`<span><span class="symbol">η</span> ${formatNumber(site.eta)}</span>`);
  if (!parts.length) return `<span class="muted">No ${symbolCq()} / η recorded</span>`;
  const confidence = site.assignment_confidence ? `<br><span class="muted">${escapeHtml(site.assignment_confidence)}</span>` : "";
  return `${parts.join(" ")}${confidence}`;
}

function siteKind(site) {
  if (site.lines.length && site.qcc_khz !== null) return "site with coupling parameters and frequency lines";
  if (site.lines.length) return site.assignment_confidence === "unassigned_frequency_list" ? "frequency list" : "frequency-bearing site";
  if (site.qcc_khz !== null || site.eta !== null) return "coupling-parameter site";
  return "site record";
}

function conditionDetails(line, sample) {
  const measurement = line.measurement || {};
  const parts = [];
  const temp = measurement.temperature_original || (line.temperature_k != null ? `${formatNumber(line.temperature_k)} K` : "");
  if (temp) parts.push(`<span><span class="symbol">T</span> ${escapeHtml(formatTemperature(temp))}</span>`);
  if (measurement.method_description || measurement.method) {
    parts.push(`<span>${escapeHtml(measurement.method_description || measurement.method)}</span>`);
  }
  if (line.form || measurement.form) parts.push(`<span>form ${escapeHtml(line.form || measurement.form)}</span>`);
  if (measurement.phase) parts.push(`<span>${escapeHtml(measurement.phase)}</span>`);
  return parts.length ? parts.join("<br>") : `<span class="muted">not recorded</span>`;
}

function lineDetails(line) {
  const parts = [];
  if (line.t1_original) parts.push(`<span><span class="symbol">T₁</span> ${escapeHtml(line.t1_original)}</span>`);
  if (line.t2_original) parts.push(`<span><span class="symbol">T₂</span> ${escapeHtml(line.t2_original)}</span>`);
  if (line.t2_star_original) parts.push(`<span><span class="symbol">T₂*</span> ${escapeHtml(line.t2_star_original)}</span>`);
  if (line.dnu_dt_original) parts.push(`<span><span class="symbol">dν/dT</span> ${escapeHtml(line.dnu_dt_original)}</span>`);
  if (line.line_width_original) parts.push(`<span>Δν ${escapeHtml(line.line_width_original)}</span>`);
  if (line.fwhm_khz !== null) parts.push(`<span>FWHM ${formatNumber(line.fwhm_khz)} kHz</span>`);
  return parts.length ? parts.join("<br>") : `<span class="muted">not recorded</span>`;
}

function measurementChips(measurement, sample) {
  const chips = [];
  const temp = measurement && (measurement.temperature_original || (sample.temperature_k != null ? `${formatNumber(sample.temperature_k)} K` : ""));
  if (temp) chips.push(chip(`T ${formatTemperature(temp)}`));
  if (measurement && (measurement.method_description || measurement.method)) chips.push(chip(measurement.method_description || measurement.method));
  if (measurement && measurement.form) chips.push(chip(`form ${measurement.form}`));
  if (measurement && measurement.phase) chips.push(chip(measurement.phase));
  return chips;
}

function assignSiteColors(samples) {
  state.siteColors = new Map();
  const palette = ["#116466", "#8a5a00", "#6f3f7a", "#2456a6", "#9b315f", "#42743b", "#6d5a2e", "#326a8a"];
  let index = 0;
  for (const sample of samples) {
    for (const site of sample.sites || []) {
      if (!state.siteColors.has(site.id)) {
        state.siteColors.set(site.id, palette[index % palette.length]);
        index += 1;
      }
    }
  }
}

function siteColor(siteId) {
  return state.siteColors.get(siteId) || "#116466";
}

function symbolCq() {
  return `<span class="symbol">C<sub>Q</sub></span>`;
}

function renderReferences(references) {
  if (!references.length) {
    els.references.innerHTML = `<div class="muted">No compound-level references. Line and site references appear in measurement provenance.</div>`;
    return;
  }
  els.references.innerHTML = references.map(ref => `
    <div class="reference-item">
      <div>${escapeHtml(ref.citation_text)}</div>
      <div class="muted">${[ref.year, ref.source_reference_number, ref.note].filter(Boolean).map(escapeHtml).join(" | ")}</div>
    </div>
  `).join("");
}

function renderSources(sources) {
  if (!sources.length) {
    els.sources.innerHTML = `<div class="muted">No source files linked.</div>`;
    return;
  }
  els.sources.innerHTML = sources.map(source => `
    <div class="source-item">
      <div>${escapeHtml(source.title)}</div>
      <div class="muted">${escapeHtml(humanSourceType(source.source_type))}</div>
      <div class="muted">${escapeHtml(source.relative_path || "")}</div>
    </div>
  `).join("");
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function chip(text, extraClass = "") {
  if (!text) return "";
  return `<span class="chip ${extraClass}">${escapeHtml(text)}</span>`;
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
  const number = Number(value);
  if (Math.abs(number) >= 1000) return number.toFixed(1).replace(/\.0$/, "");
  if (Math.abs(number) >= 10) return number.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  return number.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function formatTemperature(value) {
  if (!value) return "";
  const text = String(value).trim();
  if (/^r\.?\s*temp\.?$/i.test(text) || /^rt$/i.test(text) || /^rtemp$/i.test(text)) {
    return "room temperature";
  }
  if (/^\d+(?:\.\d+)?$/.test(text)) {
    return `${formatNumber(text)} K`;
  }
  return text;
}

function formatFormula(value) {
  if (!value) return "";
  return escapeHtml(String(value)).replace(/([A-Za-z\)])(\d+)/g, "$1<sub>$2</sub>");
}

function humanSourceType(value) {
  const labels = {
    cwru_compact_pdf: "CWRU/UF compact PDF",
    cwru_google_sites_wayback_html: "CWRU/UF archived web page",
    kcl_experimental_note_pdf: "King's College note",
    landolt_bornstein_pdf: "Landolt-Bornstein",
    nrl_nqr_data_tables_detailed_pdf: "Navy/NRL detailed table",
    nrl_nqr_data_tables_line_summary_pdf: "Navy/NRL line table",
    nrl_nqr_data_tables_site_summary_pdf: "Navy/NRL site table"
  };
  return labels[value] || value || "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function debounce(fn, delay) {
  let handle;
  return (...args) => {
    clearTimeout(handle);
    handle = setTimeout(() => fn(...args), delay);
  };
}
