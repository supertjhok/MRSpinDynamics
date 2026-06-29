function generate_optimization_result_fixtures_main()
% GENERATE_OPTIMIZATION_RESULT_FIXTURES_MAIN
% Generate compact MATLAB optimizer-output fixtures for workflow validation.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
matlab_code = resolve_matlab_code(repo_root);
fixture_dir = fullfile(repo_root, 'validation', 'fixtures');

if exist(fixture_dir, 'dir') ~= 7
    mkdir(fixture_dir);
end

set(0, 'DefaultFigureVisible', 'off');

addpath(fileparts(matlab_code));
addpath(fullfile(matlab_code, 'calc_echo'));
addpath(fullfile(matlab_code, 'calc_macq'));
addpath(fullfile(matlab_code, 'calc_masy'));
addpath(fullfile(matlab_code, 'calc_rot'));
addpath(fullfile(matlab_code, 'Params'));
addpath(fullfile(matlab_code, 'opt_pulse'));
addpath(fullfile(matlab_code, 'sim_spin_dynamics_arb'));
addpath(fullfile(matlab_code, 'sim_spin_dynamics_asymp'));
addpath(fullfile(matlab_code, 'time_varying_field'));
addpath(fullfile(matlab_code, 'circuit_simulation', 'tuned_probe'));
addpath(fullfile(matlab_code, 'circuit_simulation', 'untuned_probe'));
addpath(fullfile(matlab_code, 'circuit_simulation', 'matched_probe'));

[params, sp, pp] = set_params_tuned_SPA;
sp.numpts = 9;
sp.del_w = linspace(-10, 10, sp.numpts);
sp.plt_axis = 0;
sp.plt_tx = 0;
sp.plt_rx = 0;
sp.plt_echo = 0;

nseg = 2;
params.neff = calc_rot_axis_arba3([pi], [0], [1], sp.del_w, 0);
params.texc = pp.T_180 * 0.1 * ones(1, nseg);
params.pexc = [0.2, 1.1];
params.aexc = ones(1, nseg);

has_fmincon = exist('fmincon', 'file') == 2;
if has_fmincon
    [~, ~, snr_initial] = evaluate_tuned_excitation_fixture( ...
        params.pexc, params.neff, sp, pp);
    out_exc = opt_exc_pulse_tuned(params, sp, pp);
    [mrx_optimized, ~, snr_optimized] = evaluate_tuned_excitation_fixture( ...
        out_exc.pexc, params.neff, sp, pp);

    exc_table = [
        0, snr_initial, params.pexc;
        1, snr_optimized, out_exc.pexc
    ];
    dlmwrite( ...
        fullfile(fixture_dir, 'optimization_tuned_excitation_result.csv'), ...
        exc_table, ...
        'precision', '%.17g');
else
    disp('fmincon unavailable; using committed compact optimizer CSV fixtures.');
    exc_table = dlmread( ...
        fullfile(fixture_dir, 'optimization_tuned_excitation_result.csv'), ...
        ',');
    params.pexc = exc_table(1, 3:end);
    out_exc = struct();
    out_exc.pexc = exc_table(2, 3:end);
    snr_initial = exc_table(1, 2);
    snr_optimized = exc_table(2, 2);
    [mrx_optimized, ~, ~] = evaluate_tuned_excitation_fixture( ...
        out_exc.pexc, params.neff, sp, pp);
end

exc_results = cell(2, 1);
exc_results{1} = excitation_result_cell( ...
    params.pexc, snr_initial, params, sp, pp);
exc_results{2} = excitation_result_cell( ...
    out_exc.pexc, snr_optimized, params, sp, pp);
results = exc_results;
save_results_mat( ...
    fullfile(fixture_dir, 'optimization_tuned_excitation_results.mat'), ...
    results);

target_mrx = mrx_optimized;
target_snr = snr_optimized;
inverse_initial = mod(out_exc.pexc + pi, 2*pi);
params_inv = params;
params_inv.mrx = target_mrx;
params_inv.SNR = target_snr;
params_inv.pexc = inverse_initial;

if has_fmincon
    out_inv = opt_exc_pulse_tuned_inv(params_inv, sp, pp);
else
    inv_table = dlmread( ...
        fullfile(fixture_dir, 'optimization_tuned_inverse_result.csv'), ...
        ',');
    inverse_initial = inv_table(1, 5:end);
    out_inv = struct();
    out_inv.pexc = inv_table(2, 5:end);
end

[mrx_inv_initial, ~, snr_inv_initial] = evaluate_tuned_excitation_fixture( ...
    inverse_initial, params.neff, sp, pp);
mismatch_initial = inverse_mismatch( ...
    target_mrx, target_snr, mrx_inv_initial, snr_inv_initial, sp.del_w);
ratio_initial = residual_ratio(target_mrx, mrx_inv_initial, sp.del_w);

[mrx_inv_optimized, ~, snr_inv_optimized] = evaluate_tuned_excitation_fixture( ...
    out_inv.pexc, params.neff, sp, pp);
mismatch_optimized = inverse_mismatch( ...
    target_mrx, target_snr, mrx_inv_optimized, snr_inv_optimized, sp.del_w);
ratio_optimized = residual_ratio(target_mrx, mrx_inv_optimized, sp.del_w);

inv_table = [
    0, mismatch_initial, ratio_initial, snr_inv_initial, inverse_initial;
    1, mismatch_optimized, ratio_optimized, snr_inv_optimized, out_inv.pexc
];
dlmwrite( ...
    fullfile(fixture_dir, 'optimization_tuned_inverse_result.csv'), ...
    inv_table, ...
    'precision', '%.17g');
write_inverse_spectra_csv( ...
    fullfile(fixture_dir, 'optimization_tuned_inverse_spectra.csv'), ...
    sp.del_w, ...
    target_mrx, ...
    mrx_inv_initial, ...
    mrx_inv_optimized);

inv_results = cell(2, 1);
inv_results{1} = excitation_result_cell( ...
    inverse_initial, mismatch_initial, params_inv, sp, pp);
inv_results{2} = excitation_result_cell( ...
    out_inv.pexc, mismatch_optimized, params_inv, sp, pp);
results = inv_results;
save_results_mat( ...
    fullfile(fixture_dir, 'optimization_tuned_excitation_results_inv.mat'), ...
    results);

del_w_v0 = linspace(-4, 4, 9);
v0_tp_1 = [0.2, pi, 0.2];
v0_phi_1 = [0, 0.25, 0];
v0_amp_1 = [0, 1, 0];
[~, v0crit_1] = evaluate_v0crit_fixture(v0_tp_1, v0_phi_1, v0_amp_1, del_w_v0);
v0_tp_2 = [0.2, pi/2, pi/2, 0.2];
v0_phi_2 = [0, 0.15, 0.45, 0];
v0_amp_2 = [0, 1, 1, 0];
[~, v0crit_2] = evaluate_v0crit_fixture(v0_tp_2, v0_phi_2, v0_amp_2, del_w_v0);
v0_results = cell(2, 1);
v0_results{1} = v0crit_result_cell( ...
    v0_tp_1, v0_phi_1, v0_amp_1, v0crit_1, del_w_v0);
v0_results{2} = v0crit_result_cell( ...
    v0_tp_2, v0_phi_2, v0_amp_2, v0crit_2, del_w_v0);
results = v0_results;
save_results_mat( ...
    fullfile(fixture_dir, 'optimization_ideal_v0crit_results.mat'), ...
    results);

[tuned_ref_results] = refocusing_result_fixture('tuned');
results = tuned_ref_results;
save_results_mat( ...
    fullfile(fixture_dir, 'optimization_tuned_refocusing_results.mat'), ...
    results);

[untuned_ref_results] = refocusing_result_fixture('untuned');
results = untuned_ref_results;
save_results_mat( ...
    fullfile(fixture_dir, 'optimization_untuned_refocusing_results.mat'), ...
    results);

[matched_ref_results] = refocusing_result_fixture('matched');
results = matched_ref_results;
save_results_mat( ...
    fullfile(fixture_dir, 'optimization_matched_refocusing_results.mat'), ...
    results);
write_refocusing_result_csv( ...
    fullfile(fixture_dir, 'optimization_matched_refocusing_result.csv'), ...
    matched_ref_results);

[ideal_tv_results] = ideal_time_varying_result_fixture();
results = ideal_tv_results;
save_results_mat( ...
    fullfile(fixture_dir, 'optimization_ideal_tv_results.mat'), ...
    results);

disp(['Wrote optimization result fixtures to ', fixture_dir]);
end

function save_results_mat(path, result_cells)
    results = result_cells;
    if exist('OCTAVE_VERSION', 'builtin')
        save('-mat7-binary', path, 'results');
    else
        save(path, 'results', '-v7');
    end
end

function result = excitation_result_cell(phases, score, params, sp, pp)
    segment_fraction = 0.1;
    texc = pp.T_180 * segment_fraction * ones(1, length(phases));
    aexc = ones(1, length(phases));
    tref = [pp.T_180];
    pref = [0];
    aref = [1];
    params_out = params;
    params_out.texc = texc;
    params_out.pexc = phases;
    params_out.aexc = aexc;
    params_out.tref = tref;
    params_out.pref = pref;
    params_out.aref = aref;
    result = {texc, phases, aexc, tref, pref, aref, score, params_out, sp, pp};
end

function [neff, v0crit] = evaluate_v0crit_fixture(tp, phi, amp, del_w)
    [neff, alpha] = calc_rot_axis_arba4(tp, phi, amp, del_w, 0);
    v0crit = calc_v0crit(del_w, neff, alpha, 0);
end

function result = v0crit_result_cell(tp, phi, amp, v0crit, del_w)
    params = struct();
    params.del_w = del_w;
    params.v0crit = v0crit;
    params.description = 'compact ideal v0crit result fixture';
    sp = struct();
    sp.del_w = del_w;
    sp.plt_axis = 0;
    sp.plt_tx = 0;
    sp.plt_rx = 0;
    pp = struct();
    pp.T_90 = 25e-6;
    pp.T_180 = 2 * pp.T_90;
    pp.tref = tp;
    pp.pref = phi;
    pp.aref = amp;
    score = 1 / mean(v0crit);
    v0crit_average = mean(v0crit);
    result = {tp, phi, amp, score, v0crit_average, params, sp, pp};
end

function results = refocusing_result_fixture(probe)
    if strcmp(probe, 'tuned')
        [params, sp, pp] = set_params_tuned_SPA;
    elseif strcmp(probe, 'untuned')
        [params, sp, pp] = set_params_untuned_SPA;
    elseif strcmp(probe, 'matched')
        [sp, pp] = set_params_matched_SPA;
        params = struct();
        pp.aexc = 6;
        pp.texc = pp.T_90 / pp.aexc;
        pp.tcorr = -(2 / pi) * pp.texc;
    else
        error(['Unsupported refocusing fixture probe: ', probe]);
    end

    sp.numpts = 9;
    sp.del_w = linspace(-10, 10, sp.numpts);
    sp.plt_axis = 0;
    sp.plt_tx = 0;
    sp.plt_rx = 0;
    sp.plt_echo = 0;

    params.aexc = 6;
    params.texc = pp.T_90 / params.aexc;
    params.delt = 0.1;
    pp.tcorr = -(2 / pi) * params.texc;

    phases_1 = [0.1, 0.4];
    phases_2 = [0.2, 1.0];
    [score_1, params_1] = evaluate_refocusing_fixture(probe, params, sp, pp, phases_1);
    [score_2, params_2] = evaluate_refocusing_fixture(probe, params, sp, pp, phases_2);

    results = cell(2, 1);
    results{1} = refocusing_result_cell(params_1, score_1, sp, pp);
    results{2} = refocusing_result_cell(params_2, score_2, sp, pp);
end

function [score, params_out] = evaluate_refocusing_fixture(probe, params, sp, pp, phases)
    params_out = params;
    params_out.pref = phases;
    params_out.aref = ones(1, length(phases));
    params_out.tref = pp.T_180 * params_out.delt * ones(1, length(phases));
    if strcmp(probe, 'tuned')
        [~, ~, ~, ~, score] = plot_masy_arbref_tuned(params_out, sp, pp);
    elseif strcmp(probe, 'untuned')
        [~, ~, ~, ~, score] = plot_masy_arbref_untuned(params_out, sp, pp);
    elseif strcmp(probe, 'matched')
        pp_match = pp;
        pp_match.pref = phases;
        pp_match.delt = params_out.delt;
        [~, ~, ~, ~, score] = plot_masy_arbref_matched(sp, pp_match);
        params_out.tref = pp.T_180 * params_out.delt * ones(1, length(phases));
        params_out.pref = pp_match.pref;
        params_out.aref = ones(1, length(phases));
    else
        error(['Unsupported refocusing fixture probe: ', probe]);
    end
end

function result = refocusing_result_cell(params, score, sp, pp)
    result = {
        params.tref, ...
        params.pref, ...
        params.aref, ...
        score, ...
        params, ...
        sp, ...
        pp ...
    };
end

function write_refocusing_result_csv(path, results)
    rows = zeros(length(results), 2 + length(results{1}{2}));
    for idx = 1:length(results)
        rows(idx, :) = [idx - 1, results{idx}{4}, results{idx}{2}];
    end
    dlmwrite(path, rows, 'precision', '%.17g');
end

function write_inverse_spectra_csv(path, del_w, target_mrx, initial_mrx, optimized_mrx)
    rows = [
        del_w(:), ...
        real(target_mrx(:)), imag(target_mrx(:)), ...
        real(initial_mrx(:)), imag(initial_mrx(:)), ...
        real(optimized_mrx(:)), imag(optimized_mrx(:)) ...
    ];
    dlmwrite(path, rows, 'precision', '%.17g');
end

function results = ideal_time_varying_result_fixture()
    [sp, pp] = set_params_ideal_tv;
    sp.nx = 1;
    sp.ny = 9;
    sp.nz = 1;
    sp.maxoffs_y = 0.1;
    sp.numpts = sp.ny;
    sp.del_w = linspace(-sp.maxoffs_y, sp.maxoffs_y, sp.numpts);
    sp.del_wg = ones(1, sp.numpts);
    sp.w_1 = ones(1, sp.numpts);
    sp.rho = 1;
    sp.T1map = 1e8;
    sp.T2map = 1e8;
    sp.m0 = ones(1, sp.numpts);
    sp.mth = ones(1, sp.numpts);
    sp.T1 = 1e8 * ones(1, sp.numpts);
    sp.T2 = 1e8 * ones(1, sp.numpts);
    sp.plt_axis = 0;
    sp.plt_tx = 0;
    sp.plt_rx = 0;
    sp.plt_echo = 0;
    sp.plt_output = 0;

    pp.NE = 16;
    pp.NEmin = pp.NE;
    w_1n = (pi / 2) / pp.T_90;
    w_0max = 1.5;
    field_offsets = w_0max * sin(2 * pi * linspace(0, 1, pp.NE));
    sp.B_0t = (w_1n / sp.gamma) * field_offsets;

    tacq = w_1n * pp.tacq;
    tdw = w_1n * pp.tdw;
    nacq = round(tacq / tdw) + 1;
    params = struct();
    params.tvect = linspace(-tacq / 2, tacq / 2, nacq)';
    params.tfp = 1.9 * pi;
    params.tref = 0.1 * pi * ones(1, 2);
    params.aref = ones(1, 2);

    phases_1 = [0.1, 0.4];
    phases_2 = [0.2, 1.0];
    [score_1, params_1] = evaluate_ideal_tv_fixture(params, sp, pp, phases_1);
    [score_2, params_2] = evaluate_ideal_tv_fixture(params, sp, pp, phases_2);

    results = cell(2, 1);
    results{1} = ideal_tv_result_cell(params_1, score_1, sp, pp);
    results{2} = ideal_tv_result_cell(params_2, score_2, sp, pp);
end

function [score, params_out] = evaluate_ideal_tv_fixture(params, sp, pp, phases)
    params_out = params;
    params_out.pref = phases;
    w_1n = (pi / 2) / pp.T_90;

    pp_main = pp;
    pp_main.tref = [params_out.tfp params_out.tref params_out.tfp] / w_1n;
    pp_main.pref = [0 params_out.pref 0];
    pp_main.aref = [0 params_out.aref 0];
    [~, echo_rx, ~] = sim_cpmg_ideal_tv_final(sp, pp_main);

    pp_ref = pp_main;
    pp_ref.NE = pp.NEmin;
    sp_ref = sp;
    sp_ref.B_0t = zeros(1, pp_ref.NE);
    [~, echo_rx_ref, ~] = sim_cpmg_ideal_tv_final(sp_ref, pp_ref);
    echo_rx_ref = conj(echo_rx_ref) / sqrt(trapz(params_out.tvect, abs(echo_rx_ref).^2));
    echo_rms = trapz(params_out.tvect, echo_rx .* echo_rx_ref);
    score = real(echo_rms) / 1e4;
end

function result = ideal_tv_result_cell(params, score, sp, pp)
    result = {
        [params.tfp params.tref params.tfp], ...
        [0 params.pref 0], ...
        [0 params.aref 0], ...
        score, ...
        params, ...
        sp, ...
        pp ...
    };
end

function [mrx, masy, snr] = evaluate_tuned_excitation_fixture(phases, neff, sp, pp)
    T_90 = pp.T_90;
    B1max = (pi/2)/(T_90*sp.gamma);
    amp_zero = pp.amp_zero;
    segment_fraction = 0.1;

    texc_params = pp.T_180 * segment_fraction * ones(1, length(phases));
    aexc_params = ones(1, length(phases));

    pp_exc = pp;
    pp_exc.tref = [texc_params pp.tqs pp.trd];
    pp_exc.pref = [phases 0 0];
    pp_exc.aref = [aexc_params 0 0];
    pp_exc.Rsref = [
        pp.Rsref(2) * ones(1, length(texc_params)) ...
        pp.Rsref(3) ...
        pp.Rsref(1)
    ];

    [tvect, Icr, ~, ~] = tuned_probe_lp_Orig(sp, pp_exc);
    delt = (pi/2) * (tvect(2) - tvect(1)) / T_90;
    texc = delt * ones(1, length(tvect));
    pexc = atan2(imag(Icr), real(Icr));
    aexc = abs(Icr) * sp.sens / B1max;
    aexc(aexc < amp_zero) = 0;
    pexc(aexc == 0) = 0;

    texc = [texc -(pp.tqs + pp.trd) * (pi/2) / T_90];
    pexc = [pexc 0];
    aexc = [aexc 0];

    tacq = (pi/2) * pp.tacq(1) / T_90;
    masy = sim_spin_dynamics_asymp_mag3(texc, pexc, aexc, neff, sp.del_w, tacq);
    [mrx, snr] = tuned_probe_rx(sp, pp, masy);
    snr = snr / 1e8;
end

function val = inverse_mismatch(target_mrx, target_snr, candidate_mrx, candidate_snr, del_w)
    val = trapz(del_w, abs(target_mrx + candidate_mrx)) ...
        + 0.8 * abs(candidate_snr - target_snr);
end

function val = residual_ratio(target_mrx, candidate_mrx, del_w)
    val = trapz(del_w, abs(target_mrx + candidate_mrx)) ...
        / trapz(del_w, abs(target_mrx));
end

function matlab_code = resolve_matlab_code(repo_root)
    parent = fileparts(repo_root);
    candidates = {
        fullfile(parent, 'Version_3', 'code'), ...
        fullfile(parent, 'MATLABSpinDynamics', 'Version_3', 'code')
    };
    for idx = 1:numel(candidates)
        if exist(candidates{idx}, 'dir') == 7
            matlab_code = candidates{idx};
            return;
        end
    end
    matlab_code = candidates{end};
end
