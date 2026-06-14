% GENERATE_OPTIMIZATION_RESULT_FIXTURES
% Generate compact MATLAB optimizer-output fixtures for workflow validation.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
matlab_code = fullfile(fileparts(repo_root), 'SpinDynamicsUpdated', 'Version_2', 'code');
fixture_dir = fullfile(repo_root, 'validation', 'fixtures');

if exist(fixture_dir, 'dir') ~= 7
    mkdir(fixture_dir);
end

set(0, 'DefaultFigureVisible', 'off');

addpath(fullfile(matlab_code, 'calc_echo'));
addpath(fullfile(matlab_code, 'calc_masy'));
addpath(fullfile(matlab_code, 'calc_rot'));
addpath(fullfile(matlab_code, 'Params'));
addpath(fullfile(matlab_code, 'opt_pulse'));
addpath(fullfile(matlab_code, 'sim_spin_dynamics_asymp'));
addpath(fullfile(matlab_code, 'circuit_simulation', 'tuned_probe'));

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

[mrx_initial, ~, snr_initial] = evaluate_tuned_excitation_fixture( ...
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

target_mrx = mrx_optimized;
target_snr = snr_optimized;
inverse_initial = mod(out_exc.pexc + pi, 2*pi);
params_inv = params;
params_inv.mrx = target_mrx;
params_inv.SNR = target_snr;
params_inv.pexc = inverse_initial;

[mrx_inv_initial, ~, snr_inv_initial] = evaluate_tuned_excitation_fixture( ...
    inverse_initial, params.neff, sp, pp);
mismatch_initial = inverse_mismatch(target_mrx, target_snr, mrx_inv_initial, snr_inv_initial, sp.del_w);
ratio_initial = residual_ratio(target_mrx, mrx_inv_initial, sp.del_w);

out_inv = opt_exc_pulse_tuned_inv(params_inv, sp, pp);
[mrx_inv_optimized, ~, snr_inv_optimized] = evaluate_tuned_excitation_fixture( ...
    out_inv.pexc, params.neff, sp, pp);
mismatch_optimized = inverse_mismatch(target_mrx, target_snr, mrx_inv_optimized, snr_inv_optimized, sp.del_w);
ratio_optimized = residual_ratio(target_mrx, mrx_inv_optimized, sp.del_w);

inv_table = [
    0, mismatch_initial, ratio_initial, snr_inv_initial, inverse_initial;
    1, mismatch_optimized, ratio_optimized, snr_inv_optimized, out_inv.pexc
];
dlmwrite( ...
    fullfile(fixture_dir, 'optimization_tuned_inverse_result.csv'), ...
    inv_table, ...
    'precision', '%.17g');

disp(['Wrote optimization result fixtures to ', fixture_dir]);

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
