// Static replacement for the explorer server's /api endpoints, used by the
// GitHub Pages build (scripts/build_static_explorer.py). The build writes
// data/stats.json, data/options.json, data/index.json (one search row per
// compound plus the fields searched) and data/compounds/<file>.json.
// searchRows mirrors explorer_server.search_compounds; the Python test
// tests/test_static_explorer.py runs both on the same queries.

(function (root) {
  const DEFAULT_LIMIT = 80;
  const MAX_LIMIT = 300;

  function parseNumber(value) {
    if (value === null || value === undefined || String(value).trim() === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  // SQLite LIKE folds ASCII case only; this also folds other letters, which
  // can only add matches for non-ASCII queries.
  function contains(value, needle) {
    return typeof value === "string" && value.toLowerCase().includes(needle);
  }

  function compareRows(a, b) {
    if (b.line_count !== a.line_count) return b.line_count - a.line_count;
    // SQLite's default BINARY collation compares code units, as JS < does.
    if (a.canonical_name < b.canonical_name) return -1;
    if (a.canonical_name > b.canonical_name) return 1;
    return 0;
  }

  // params: anything with get(name) -> string | null, e.g. URLSearchParams.
  function searchRows(index, params) {
    const text = (params.get("q") || "").toLowerCase();
    const category = params.get("category") || "";
    const isotope = params.get("isotope") || "";
    const sourceType = params.get("source_type") || "";
    const freqMin = parseNumber(params.get("freq_min"));
    const freqMax = parseNumber(params.get("freq_max"));
    const requested = parseInt(params.get("limit") || String(DEFAULT_LIMIT), 10);
    const limit = Math.max(1, Math.min(Number.isNaN(requested) ? DEFAULT_LIMIT : requested, MAX_LIMIT));

    const rows = index.filter(entry => {
      if (text) {
        const fields = [
          entry.row.canonical_name,
          entry.row.formula,
          entry.row.conventional_formula,
          entry.notes,
          ...entry.aliases
        ];
        if (!fields.some(field => contains(field, text))) return false;
      }
      if (category && entry.row.category !== category) return false;
      if (isotope && !entry.row.isotopes.includes(isotope)) return false;
      if (sourceType && !entry.row.source_types.includes(sourceType)) return false;
      if (freqMin !== null || freqMax !== null) {
        const inRange = entry.frequencies_khz.some(
          frequency => (freqMin === null || frequency >= freqMin) && (freqMax === null || frequency <= freqMax)
        );
        if (!inRange) return false;
      }
      return true;
    }).map(entry => entry.row);
    rows.sort(compareRows);
    const limited = rows.slice(0, limit);
    return { rows: limited, count: limited.length };
  }

  function createApi(fetchJson) {
    let indexPromise = null;
    let filesById = null;
    const loadIndex = () => {
      if (!indexPromise) {
        indexPromise = fetchJson("data/index.json").then(index => {
          filesById = new Map(index.map(entry => [entry.row.id, entry.file]));
          return index;
        });
      }
      return indexPromise;
    };
    return {
      stats: () => fetchJson("data/stats.json"),
      options: () => fetchJson("data/options.json"),
      search: async params => searchRows(await loadIndex(), params),
      compound: async id => {
        await loadIndex();
        const file = filesById.get(id);
        if (!file) throw new Error(`unknown compound ${id}`);
        return fetchJson(`data/compounds/${file}`);
      }
    };
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { searchRows };
  } else {
    root.NQR_STATIC_API = createApi(async url => {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return response.json();
    });
  }
})(typeof window !== "undefined" ? window : globalThis);
