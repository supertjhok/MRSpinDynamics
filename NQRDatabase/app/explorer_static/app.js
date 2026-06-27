const state = {
  options: null,
  results: [],
  selectedId: null,
  selected: null
};

const els = {
  stats: document.getElementById("stats"),
  searchForm: document.getElementById("searchForm"),
  query: document.getElementById("query"),
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
    `${counts.lines || 0} lines`,
    `${counts.references || 0} references`,
    `${counts.sources || 0} sources`
  ].join("  ");
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
  } else if (!state.results.some(row => row.id === state.selectedId)) {
    await selectCompound(state.results[0].id);
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
  els.emptyState.classList.remove("hidden");
  els.detail.classList.add("hidden");
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
    ...(compound.aliases || []).slice(0, 6).map(alias => chip(alias))
  ].join("");
  const searchUrl = compound.structure && compound.structure.pubchem_search_url;
  els.pubchemLink.href = searchUrl || "https://pubchem.ncbi.nlm.nih.gov/";
  els.pubchemLink.style.visibility = searchUrl ? "visible" : "hidden";
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
  for (const point of points.slice(0, 180)) {
    const line = document.createElement("div");
    const left = 4 + ((point.frequency_khz - min) / span) * 92;
    line.className = "spectrum-line";
    line.style.left = `${left}%`;
    line.style.height = `${38 + Math.min(120, (point.frequency_khz - min) / span * 90)}px`;
    line.dataset.label = `${formatNumber(point.frequency_khz)} ${point.isotope || ""}`;
    line.title = [
      `${formatNumber(point.frequency_khz)} kHz`,
      point.isotope,
      point.site_label,
      point.sample_label,
      point.source_type && humanSourceType(point.source_type)
    ].filter(Boolean).join(" | ");
    els.spectrumPlot.appendChild(line);
  }
  addAxisLabel(`${formatNumber(min)} kHz`, "8px");
  addAxisLabel(`${formatNumber(max)} kHz`, "calc(100% - 84px)");
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
    const rows = [];
    for (const site of sample.sites) {
      if (site.lines.length) {
        for (const line of site.lines) {
          rows.push(lineRow(sample, site, line));
        }
      } else {
        rows.push(siteOnlyRow(sample, site));
      }
    }
    group.innerHTML = `
      <div class="sample-title">
        <span>${escapeHtml(sample.label)}</span>
        <span class="muted">${sample.temperature_k !== null ? `${formatNumber(sample.temperature_k)} K` : ""}</span>
      </div>
      <table class="line-table">
        <thead>
          <tr>
            <th>Isotope / site</th>
            <th>Frequency</th>
            <th>Q.C.C. / eta</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>${rows.join("")}</tbody>
      </table>
    `;
    els.measurements.appendChild(group);
  }
}

function lineRow(sample, site, line) {
  return `
    <tr>
      <td>${escapeHtml(site.isotope || "")}<br><span class="muted">${escapeHtml(site.site_label || "")}</span></td>
      <td>${formatNumber(line.frequency_khz)} kHz<br><span class="muted">${escapeHtml(line.frequency_original || "")}</span></td>
      <td>${formatQccEta(site)}</td>
      <td>${escapeHtml(humanSourceType(line.source_type || site.source_type || ""))}</td>
    </tr>
  `;
}

function siteOnlyRow(sample, site) {
  return `
    <tr>
      <td>${escapeHtml(site.isotope || "")}<br><span class="muted">${escapeHtml(site.site_label || "")}</span></td>
      <td><span class="muted">unassigned</span></td>
      <td>${formatQccEta(site)}</td>
      <td>${escapeHtml(humanSourceType(site.source_type || ""))}</td>
    </tr>
  `;
}

function formatQccEta(site) {
  const parts = [];
  if (site.qcc_khz !== null) parts.push(`${formatNumber(site.qcc_khz)} kHz`);
  if (site.eta !== null) parts.push(`eta ${formatNumber(site.eta)}`);
  if (!parts.length) return `<span class="muted">not recorded</span>`;
  const confidence = site.assignment_confidence ? `<br><span class="muted">${escapeHtml(site.assignment_confidence)}</span>` : "";
  return `${parts.join(" / ")}${confidence}`;
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
