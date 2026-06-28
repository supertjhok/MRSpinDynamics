const fieldDefs = [
  ["substance_name", "Name"],
  ["formula_raw", "Formula"],
  ["cas_registry_number", "CAS"],
  ["nucleus", "Nucleus"],
  ["method", "Method"],
  ["temperature_original", "Temp."],
  ["frequencies_raw", "Frequencies (list)"],
  ["qcc_original", "Q.C.C. (list)"],
  ["eta_original", "eta (list)"],
  ["reference_code", "Ref."]
];

const state = {
  rows: [],
  selectedId: null,
  selected: null,
  status: "unreviewed",
  priority: "all",
  q: "",
  zoom: 1,
  pendingStatus: "unreviewed"
};

const els = {
  counts: document.getElementById("counts"),
  search: document.getElementById("search"),
  refresh: document.getElementById("refresh"),
  statusFilters: document.getElementById("statusFilters"),
  priorityFilters: document.getElementById("priorityFilters"),
  queue: document.getElementById("queue"),
  itemTitle: document.getElementById("itemTitle"),
  cropImage: document.getElementById("cropImage"),
  rawText: document.getElementById("rawText"),
  imageWrap: document.querySelector(".image-wrap"),
  consistencyBanner: document.getElementById("consistencyBanner"),
  fields: document.getElementById("fields"),
  measurementSets: document.getElementById("measurementSets"),
  addMeasurementSet: document.getElementById("addMeasurementSet"),
  notes: document.getElementById("notes"),
  reviewForm: document.getElementById("reviewForm"),
  save: document.getElementById("save"),
  saveState: document.getElementById("saveState"),
  zoomOut: document.getElementById("zoomOut"),
  zoomIn: document.getElementById("zoomIn"),
  zoomLabel: document.getElementById("zoomLabel")
};

init();

function init() {
  renderFilters();
  els.refresh.addEventListener("click", loadQueue);
  els.search.addEventListener("input", debounce(() => {
    state.q = els.search.value.trim();
    loadQueue();
  }, 180));
  els.reviewForm.addEventListener("submit", saveCurrent);
  document.querySelectorAll(".status-actions button").forEach(button => {
    button.addEventListener("click", () => {
      state.pendingStatus = button.dataset.status;
      renderStatusButtons();
    });
  });
  els.zoomOut.addEventListener("click", () => setZoom(state.zoom - 0.15));
  els.zoomIn.addEventListener("click", () => setZoom(state.zoom + 0.15));
  els.addMeasurementSet.addEventListener("click", addMeasurementSet);
  loadQueue();
}

function renderFilters() {
  const statuses = [
    ["all", "All"],
    ["unreviewed", "Unreviewed"],
    ["accepted", "Accepted"],
    ["needs_manual_fix", "Needs Fix"],
    ["rejected", "Rejected"]
  ];
  const priorities = [
    ["all", "P All"],
    ["1", "P1"],
    ["2", "P2"],
    ["3", "P3"]
  ];
  renderFilterButtons(els.statusFilters, statuses, "status");
  renderFilterButtons(els.priorityFilters, priorities, "priority");
}

function renderFilterButtons(container, items, key) {
  container.innerHTML = "";
  for (const [value, label] of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.className = state[key] === value ? "active" : "";
    button.addEventListener("click", () => {
      state[key] = value;
      renderFilters();
      loadQueue();
    });
    container.appendChild(button);
  }
}

async function loadQueue() {
  const params = new URLSearchParams();
  params.set("status", state.status);
  params.set("priority", state.priority);
  if (state.q) params.set("q", state.q);
  const data = await getJson(`/api/queue?${params.toString()}`);
  state.rows = data.rows;
  renderCounts(data.counts);
  renderQueue();
  if (!state.rows.length) {
    clearDetail();
    return;
  }
  if (!state.rows.some(row => row.id === state.selectedId)) {
    await selectItem(state.rows[0].id);
  } else {
    markSelected();
  }
}

function renderCounts(counts) {
  const status = counts.status || {};
  const priority = counts.priority || {};
  els.counts.textContent = [
    `Unreviewed ${status.unreviewed || 0}`,
    `Accepted ${status.accepted || 0}`,
    `Fix ${status.needs_manual_fix || 0}`,
    `Rejected ${status.rejected || 0}`,
    `P1 ${priority["1"] || 0}`
  ].join("  ");
}

function renderQueue() {
  els.queue.innerHTML = "";
  for (const row of state.rows) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "queue-item";
    button.dataset.id = row.id;
    button.innerHTML = `
      <div class="queue-heading">
        <span>Table ${escapeHtml(row.table_number)} / ${escapeHtml(row.substance_number)}</span>
        <span class="pill p${row.priority}">P${row.priority}</span>
      </div>
      <div class="queue-name">${escapeHtml(row.substance_name || row.formula_raw || row.entry_id)}</div>
      <div class="meta-row">
        ${pill(row.status)}
        ${row.reference_code ? pill(row.reference_code) : ""}
        ${row.cas_registry_number ? pill(row.cas_registry_number) : ""}
      </div>
      <div class="flag-row">${(row.issue_flags || []).slice(0, 3).map(pill).join("")}</div>
    `;
    button.addEventListener("click", () => selectItem(row.id));
    els.queue.appendChild(button);
  }
  markSelected();
}

async function selectItem(id) {
  state.selectedId = id;
  state.selected = await getJson(`/api/item/${encodeURIComponent(id)}`);
  state.pendingStatus = state.selected.status || "unreviewed";
  state.zoom = 1;
  renderDetail();
  markSelected();
}

function markSelected() {
  document.querySelectorAll(".queue-item").forEach(item => {
    item.classList.toggle("active", item.dataset.id === state.selectedId);
  });
}

function renderDetail() {
  const item = state.selected;
  if (!item) {
    clearDetail();
    return;
  }
  els.itemTitle.textContent = `Table ${item.table_number} / ${item.substance_number}  ${item.substance_name || item.formula_raw || ""}`;
  if (item.crop_relative_path) {
    els.cropImage.src = cropUrl(item.crop_relative_path);
    els.cropImage.onload = fitImageToWidth;
    els.cropImage.style.display = "block";
  } else {
    els.cropImage.removeAttribute("src");
    els.cropImage.style.display = "none";
  }
  setZoom(1);
  els.rawText.textContent = [
    "Measurement sets are separated by method, temperature, and reference. Frequency records and Q.C.C./eta records are independent lists within each set; no element-wise assignment is inferred.",
    roomTemperatureNote(item.temperature_original),
    "",
    item.raw_table_text || "",
    "",
    item.raw_footnote_text || "",
    "",
    (item.issue_flags || []).join(", ")
  ].join("\n").trim();
  renderConsistencyBanner(item.consistency);
  renderFields(item);
  renderMeasurementSets(item);
  els.notes.value = item.reviewer_notes || "";
  renderStatusButtons();
  els.saveState.textContent = "";
}

function roomTemperatureNote(value) {
  if (!value) return "";
  const token = String(value).replaceAll(".", "").trim().toLowerCase();
  return ["rt", "rtemp"].includes(token)
    ? `${value} in the temperature field means room temperature; no exact numeric value is implied.`
    : "";
}

function renderConsistencyBanner(consistency) {
  if (!els.consistencyBanner) return;
  if (!consistency || Number(consistency.flagged) !== 1) {
    els.consistencyBanner.innerHTML = "";
    return;
  }
  const gapKhz = Math.round((consistency.max_gap_hz || 0) / 1e3);
  const predicted = parseJsonList(consistency.predicted_strong_mhz);
  const implied = predicted.length
    ? `<div class="consistency-implied">QCC/η predict strong lines near ${predicted.map(v => v + " MHz").join(", ")}</div>`
    : "";
  els.consistencyBanner.innerHTML = `
    <div class="consistency-banner flagged">
      <span class="consistency-icon">⚠</span>
      <div>
        <div class="consistency-heading">Simulator check: lines vs QCC/η disagree (${gapKhz} kHz)</div>
        <div class="consistency-detail">${escapeHtml(consistency.detail || "")}</div>
        ${implied}
      </div>
    </div>
  `;
}

function parseJsonList(value) {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    return [];
  }
}

function renderFields(item) {
  els.fields.innerHTML = "";
  for (const [name, label] of fieldDefs) {
    const wrap = document.createElement("div");
    wrap.className = "field";
    const inputId = `field_${name}`;
    wrap.innerHTML = `
      <label for="${inputId}">${label}</label>
      <input id="${inputId}" name="${name}" value="${escapeAttr(item[name] || "")}">
    `;
    els.fields.appendChild(wrap);
  }
}

function renderMeasurementSets(item) {
  els.measurementSets.innerHTML = "";
  const sets = item.measurement_sets && item.measurement_sets.length
    ? item.measurement_sets
    : [{
      set_index: 1,
      method: item.method || "",
      method_description: "",
      temperature_original: item.temperature_original || "",
      reference_code: item.reference_code || "",
      remark_flag: item.remark_flag || "",
      raw_set_text: item.raw_table_text || "",
      notes: "",
      frequency_records: item.frequency_records || [],
      qcc_eta_records: item.qcc_eta_records || []
    }];
  sets.forEach((measurementSet, index) => {
    els.measurementSets.appendChild(measurementSetNode(measurementSet, index));
  });
}

function measurementSetNode(measurementSet, index) {
  const card = document.createElement("div");
  card.className = "site-card measurement-set-card";
  card.dataset.index = index;
  card.innerHTML = `
    <div class="site-card-header">
      <div class="site-card-title">Set ${index + 1}</div>
      <button type="button" class="remove-set">Remove</button>
    </div>
    <div class="site-fields">
      <div class="field">
        <label>Method</label>
        <input data-set-field="method" value="${escapeAttr(measurementSet.method || "")}">
      </div>
      <div class="field">
        <label>Temp.</label>
        <input data-set-field="temperature_original" value="${escapeAttr(measurementSet.temperature_original || "")}">
      </div>
      <div class="field">
        <label>Ref.</label>
        <input data-set-field="reference_code" value="${escapeAttr(measurementSet.reference_code || "")}">
      </div>
      <div class="field">
        <label>Remark</label>
        <input data-set-field="remark_flag" value="${escapeAttr(measurementSet.remark_flag || "")}">
      </div>
      <div class="field wide">
        <label>Set Notes</label>
        <input data-set-field="notes" value="${escapeAttr(measurementSet.notes || "")}">
      </div>
      <input type="hidden" data-set-field="method_description" value="${escapeAttr(measurementSet.method_description || "")}">
      <input type="hidden" data-set-field="raw_set_text" value="${escapeAttr(measurementSet.raw_set_text || "")}">
    </div>
    ${measurementSet.method_description ? `<div class="method-note">${escapeHtml(measurementSet.method_description)}</div>` : ""}
    <div class="subrecord-section">
      <div class="subrecord-heading">
        <h3>Frequencies</h3>
        <button type="button" class="add-frequency">Add Frequency</button>
      </div>
      <div class="record-list frequency-records"></div>
    </div>
    <div class="subrecord-section">
      <div class="subrecord-heading">
        <h3>Q.C.C. / eta Pairs</h3>
        <button type="button" class="add-qcc-eta">Add Q.C.C./eta</button>
      </div>
      <div class="record-list qcc-eta-records"></div>
    </div>
  `;
  const frequencyRecords = card.querySelector(".frequency-records");
  const qccEtaRecords = card.querySelector(".qcc-eta-records");
  (measurementSet.frequency_records || []).forEach((record, recordIndex) => {
    frequencyRecords.appendChild(frequencyRecordRow(record, recordIndex));
  });
  (measurementSet.qcc_eta_records || []).forEach((record, recordIndex) => {
    qccEtaRecords.appendChild(qccEtaRecordRow(record, recordIndex));
  });
  card.querySelector(".remove-set").addEventListener("click", () => {
    card.remove();
    renumberMeasurementSets();
  });
  card.querySelector(".add-frequency").addEventListener("click", () => {
    const recordIndex = frequencyRecords.querySelectorAll(".record-row").length;
    frequencyRecords.appendChild(frequencyRecordRow({frequency_original: "", notes: ""}, recordIndex));
  });
  card.querySelector(".add-qcc-eta").addEventListener("click", () => {
    const recordIndex = qccEtaRecords.querySelectorAll(".record-row").length;
    qccEtaRecords.appendChild(qccEtaRecordRow({qcc_original: "", eta_original: "", notes: ""}, recordIndex));
  });
  return card;
}

function frequencyRecordRow(record, index) {
  const row = document.createElement("div");
  row.className = "record-row frequency-row";
  row.dataset.index = index;
  row.innerHTML = `
    <div class="record-label">Frequency ${index + 1}</div>
    <input data-frequency-field="frequency_original" value="${escapeAttr(record.frequency_original || "")}">
    <input data-frequency-field="notes" value="${escapeAttr(record.notes || "")}" placeholder="Notes">
    <button type="button" class="remove-record">Remove</button>
  `;
  row.querySelector(".remove-record").addEventListener("click", () => {
    const container = row.parentElement;
    row.remove();
    renumberChildRecords(container, "Frequency");
  });
  return row;
}

function qccEtaRecordRow(record, index) {
  const row = document.createElement("div");
  row.className = "record-row qcc-eta-row";
  row.dataset.index = index;
  row.innerHTML = `
    <div class="record-label">Q.C.C./eta ${index + 1}</div>
    <input data-qcc-eta-field="qcc_original" value="${escapeAttr(record.qcc_original || "")}" placeholder="Q.C.C.">
    <input data-qcc-eta-field="eta_original" value="${escapeAttr(record.eta_original || "")}" placeholder="eta">
    <input data-qcc-eta-field="notes" value="${escapeAttr(record.notes || "")}" placeholder="Notes">
    <button type="button" class="remove-record">Remove</button>
  `;
  row.querySelector(".remove-record").addEventListener("click", () => {
    const container = row.parentElement;
    row.remove();
    renumberChildRecords(container, "Q.C.C./eta");
  });
  return row;
}

function addMeasurementSet() {
  const index = els.measurementSets.querySelectorAll(".measurement-set-card").length;
  els.measurementSets.appendChild(measurementSetNode({
    set_index: index + 1,
    method: "",
    method_description: "",
    temperature_original: "",
    reference_code: "",
    remark_flag: "",
    raw_set_text: "",
    notes: "",
    frequency_records: [],
    qcc_eta_records: []
  }, index));
}

function collectMeasurementSets() {
  return Array.from(els.measurementSets.querySelectorAll(".measurement-set-card")).map((card, index) => {
    const setValue = name => card.querySelector(`[data-set-field="${name}"]`)?.value.trim() || "";
    return {
      set_index: index + 1,
      method: setValue("method"),
      method_description: setValue("method_description"),
      temperature_original: setValue("temperature_original"),
      reference_code: setValue("reference_code"),
      remark_flag: setValue("remark_flag"),
      raw_set_text: setValue("raw_set_text"),
      notes: setValue("notes"),
      frequency_records: collectFrequencyRows(card),
      qcc_eta_records: collectQccEtaRows(card)
    };
  });
}

function collectFrequencyRows(card) {
  return Array.from(card.querySelectorAll(".frequency-records .record-row")).map((row, index) => {
    const value = name => row.querySelector(`[data-frequency-field="${name}"]`)?.value.trim() || "";
    return {
      sequence_index: index + 1,
      frequency_original: value("frequency_original"),
      notes: value("notes")
    };
  });
}

function collectQccEtaRows(card) {
  return Array.from(card.querySelectorAll(".qcc-eta-records .record-row")).map((row, index) => {
    const value = name => row.querySelector(`[data-qcc-eta-field="${name}"]`)?.value.trim() || "";
    return {
      sequence_index: index + 1,
      qcc_original: value("qcc_original"),
      eta_original: value("eta_original"),
      notes: value("notes")
    };
  });
}

function renumberMeasurementSets() {
  els.measurementSets.querySelectorAll(".measurement-set-card").forEach((card, index) => {
    card.dataset.index = index;
    card.querySelector(".site-card-title").textContent = `Set ${index + 1}`;
  });
}

function renumberChildRecords(container, label) {
  if (!container) return;
  container.querySelectorAll(".record-row").forEach((row, index) => {
    row.dataset.index = index;
    row.querySelector(".record-label").textContent = `${label} ${index + 1}`;
  });
}

function renderStatusButtons() {
  document.querySelectorAll(".status-actions button").forEach(button => {
    button.classList.toggle("active", button.dataset.status === state.pendingStatus);
  });
}

async function saveCurrent(event) {
  event.preventDefault();
  if (!state.selected) return;
  const fieldEdits = {};
  for (const [name] of fieldDefs) {
    const input = document.getElementById(`field_${name}`);
    fieldEdits[name] = input.value.trim();
  }
  els.save.disabled = true;
  els.saveState.textContent = "Saving";
  try {
    const saved = await postJson(`/api/review/${encodeURIComponent(state.selected.id)}`, {
      status: state.pendingStatus,
      reviewer_notes: els.notes.value.trim(),
      field_edits: fieldEdits,
      measurement_sets: collectMeasurementSets()
    });
    state.selected = saved;
    els.saveState.textContent = "Saved";
    await loadQueue();
  } catch (error) {
    els.saveState.textContent = error.message;
  } finally {
    els.save.disabled = false;
  }
}

function clearDetail() {
  state.selected = null;
  state.selectedId = null;
  els.itemTitle.textContent = "No row selected";
  els.cropImage.removeAttribute("src");
  els.cropImage.style.display = "none";
  els.rawText.textContent = "";
  els.fields.innerHTML = "";
  els.measurementSets.innerHTML = "";
  els.notes.value = "";
}

function setZoom(value) {
  state.zoom = Math.min(2.5, Math.max(0.35, value));
  if (els.cropImage.naturalWidth) {
    els.cropImage.style.width = `${Math.round(els.cropImage.naturalWidth * state.zoom)}px`;
    els.cropImage.style.height = "auto";
  }
  els.zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
}

function fitImageToWidth() {
  if (!els.cropImage.naturalWidth || !els.imageWrap) return;
  const available = Math.max(260, els.imageWrap.clientWidth - 28);
  const fitted = available / els.cropImage.naturalWidth;
  setZoom(Math.min(1, Math.max(0.35, fitted)));
}

function cropUrl(relativePath) {
  return `/crops/${relativePath.split("/").map(encodeURIComponent).join("/")}`;
}

function splitList(value) {
  if (!value) return [];
  return String(value)
    .split(/[;\n,]+/)
    .map(part => part.trim())
    .filter(Boolean);
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Request failed: ${response.status}`);
  return response.json();
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed: ${response.status}`);
  return data;
}

function pill(value) {
  return `<span class="pill">${escapeHtml(String(value).replaceAll("_", " "))}</span>`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[char]));
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#96;");
}

function debounce(fn, wait) {
  let timer = null;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), wait);
  };
}
