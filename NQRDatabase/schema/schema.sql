PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    relative_path TEXT,
    url TEXT,
    captured_at TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS compounds (
    id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    formula TEXT,
    conventional_formula TEXT,
    category TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS compound_aliases (
    compound_id TEXT NOT NULL REFERENCES compounds(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    PRIMARY KEY (compound_id, alias)
);

CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    compound_id TEXT NOT NULL REFERENCES compounds(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    form TEXT,
    phase TEXT,
    temperature_k REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS sites (
    id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    site_number TEXT,
    isotope TEXT,
    site_label TEXT,
    weight_percent REAL,
    qcc_khz REAL,
    eta REAL,
    assignment_confidence TEXT,
    source_id TEXT REFERENCES sources(id),
    original_record TEXT
);

CREATE TABLE IF NOT EXISTS lines (
    id TEXT PRIMARY KEY,
    site_id TEXT NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
    frequency_khz REAL,
    frequency_original TEXT,
    transition_label TEXT,
    fwhm_khz REAL,
    line_width_khz REAL,
    line_width_original TEXT,
    t1_s REAL,
    t1_original TEXT,
    t2_s REAL,
    t2_original TEXT,
    t2_star_s REAL,
    t2_star_original TEXT,
    dnu_dt_khz_per_c REAL,
    dnu_dt_original TEXT,
    polarization_factor REAL,
    polarization_factor_original TEXT,
    temperature_k REAL,
    form TEXT,
    source_id TEXT REFERENCES sources(id),
    original_record TEXT
);

CREATE TABLE IF NOT EXISTS pulse_responses (
    id TEXT PRIMARY KEY,
    line_id TEXT NOT NULL REFERENCES lines(id) ON DELETE CASCADE,
    method TEXT NOT NULL,
    tau_ms REAL,
    topt_s REAL,
    i0 REAL,
    iave REAL,
    imax REAL,
    imin REAL,
    along REAL,
    tlong_s REAL,
    tshort_s REAL,
    source_id TEXT REFERENCES sources(id),
    original_record TEXT
);

CREATE TABLE IF NOT EXISTS literature_references (
    id TEXT PRIMARY KEY,
    citation_text TEXT NOT NULL,
    reference_type TEXT,
    authors TEXT,
    year INTEGER,
    title TEXT,
    journal TEXT,
    doi TEXT,
    source_id TEXT REFERENCES sources(id),
    source_page TEXT,
    source_reference_number TEXT,
    original_text TEXT
);

CREATE TABLE IF NOT EXISTS reference_links (
    id TEXT PRIMARY KEY,
    reference_id TEXT NOT NULL REFERENCES literature_references(id) ON DELETE CASCADE,
    compound_id TEXT REFERENCES compounds(id) ON DELETE CASCADE,
    site_id TEXT REFERENCES sites(id) ON DELETE CASCADE,
    line_id TEXT REFERENCES lines(id) ON DELETE CASCADE,
    source_id TEXT REFERENCES sources(id),
    link_type TEXT NOT NULL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS nqr_transition_equations (
    id TEXT PRIMARY KEY,
    nuclear_spin TEXT NOT NULL,
    transition_label TEXT,
    expression_text TEXT NOT NULL,
    expression_latex TEXT,
    variables_json TEXT,
    source_id TEXT REFERENCES sources(id),
    source_page TEXT,
    confidence TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS landolt_column_definitions (
    id TEXT PRIMARY KEY,
    column_name TEXT NOT NULL,
    definition_text TEXT NOT NULL,
    source_id TEXT REFERENCES sources(id),
    source_page TEXT
);

CREATE TABLE IF NOT EXISTS landolt_page_extracts (
    id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES sources(id),
    source_page INTEGER,
    extraction_method TEXT,
    text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS landolt_compound_entries (
    id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES sources(id),
    source_page INTEGER,
    table_number TEXT,
    substance_number TEXT,
    formula_raw TEXT,
    nucleus TEXT,
    method TEXT,
    temperature_original TEXT,
    frequencies_raw TEXT,
    qcc_original TEXT,
    eta_original TEXT,
    reference_code TEXT,
    remark_flag TEXT,
    substance_name TEXT,
    cas_registry_number TEXT,
    raw_table_text TEXT,
    raw_footnote_text TEXT,
    extraction_confidence TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS landolt_reference_codes (
    id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES sources(id),
    table_number TEXT,
    reference_code TEXT NOT NULL,
    citation_text TEXT NOT NULL,
    source_page TEXT,
    extraction_confidence TEXT
);

CREATE TABLE IF NOT EXISTS landolt_measurement_sets (
    id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES landolt_compound_entries(id) ON DELETE CASCADE,
    source_id TEXT REFERENCES sources(id),
    source_page INTEGER,
    table_number TEXT,
    substance_number TEXT,
    set_index INTEGER NOT NULL,
    method TEXT,
    method_description TEXT,
    temperature_original TEXT,
    reference_code TEXT,
    remark_flag TEXT,
    raw_set_text TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS landolt_frequency_records (
    id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES landolt_compound_entries(id) ON DELETE CASCADE,
    measurement_set_id TEXT REFERENCES landolt_measurement_sets(id) ON DELETE CASCADE,
    source_id TEXT REFERENCES sources(id),
    source_page INTEGER,
    table_number TEXT,
    substance_number TEXT,
    sequence_index INTEGER NOT NULL,
    frequency_original TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS landolt_qcc_eta_records (
    id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES landolt_compound_entries(id) ON DELETE CASCADE,
    measurement_set_id TEXT REFERENCES landolt_measurement_sets(id) ON DELETE CASCADE,
    source_id TEXT REFERENCES sources(id),
    source_page INTEGER,
    table_number TEXT,
    substance_number TEXT,
    sequence_index INTEGER NOT NULL,
    qcc_original TEXT,
    eta_original TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS landolt_review_queue (
    id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES landolt_compound_entries(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL,
    issue_flags_json TEXT,
    crop_relative_path TEXT,
    crop_bbox_json TEXT,
    source_id TEXT REFERENCES sources(id),
    source_page INTEGER,
    reviewer_notes TEXT,
    promoted_line_id TEXT REFERENCES lines(id),
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_compounds_name ON compounds(canonical_name);
CREATE INDEX IF NOT EXISTS idx_compounds_category ON compounds(category);
CREATE INDEX IF NOT EXISTS idx_sites_isotope ON sites(isotope);
CREATE INDEX IF NOT EXISTS idx_lines_frequency ON lines(frequency_khz);
CREATE INDEX IF NOT EXISTS idx_lines_source ON lines(source_id);
CREATE INDEX IF NOT EXISTS idx_literature_references_year ON literature_references(year);
CREATE INDEX IF NOT EXISTS idx_reference_links_compound ON reference_links(compound_id);
CREATE INDEX IF NOT EXISTS idx_reference_links_line ON reference_links(line_id);
CREATE INDEX IF NOT EXISTS idx_landolt_entries_substance ON landolt_compound_entries(substance_number);
CREATE INDEX IF NOT EXISTS idx_landolt_entries_formula ON landolt_compound_entries(formula_raw);
CREATE INDEX IF NOT EXISTS idx_landolt_reference_codes_code ON landolt_reference_codes(reference_code);
CREATE INDEX IF NOT EXISTS idx_landolt_measurement_entry ON landolt_measurement_sets(entry_id);
CREATE INDEX IF NOT EXISTS idx_landolt_frequency_entry ON landolt_frequency_records(entry_id);
CREATE INDEX IF NOT EXISTS idx_landolt_qcc_eta_entry ON landolt_qcc_eta_records(entry_id);
CREATE INDEX IF NOT EXISTS idx_landolt_review_status ON landolt_review_queue(status);
CREATE INDEX IF NOT EXISTS idx_landolt_review_priority ON landolt_review_queue(priority);
