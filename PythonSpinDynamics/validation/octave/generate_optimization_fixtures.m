% GENERATE_OPTIMIZATION_FIXTURES
% Generate compact MATLAB reference CSV files for optimization parity checks.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
matlab_code = resolve_matlab_code(repo_root);
fixture_dir = fullfile(repo_root, 'validation', 'fixtures');

if exist(fixture_dir, 'dir') ~= 7
    mkdir(fixture_dir);
end

addpath(fullfile(matlab_code, 'calc_echo'));
addpath(fullfile(matlab_code, 'calc_masy'));
addpath(fullfile(matlab_code, 'calc_rot'));
addpath(fullfile(matlab_code, 'Params'));
addpath(fullfile(matlab_code, 'sim_spin_dynamics_asymp'));
addpath(fullfile(matlab_code, 'circuit_simulation', 'tuned_probe'));

[~, sp, pp] = set_params_tuned_SPA;
sp.numpts = 17;
sp.del_w = linspace(-10, 10, sp.numpts);
sp.plt_axis = 0;
sp.plt_tx = 0;
sp.plt_rx = 0;

neff = calc_rot_axis_arba3([pi], [0], [1], sp.del_w, 0);
phases = [0.2, 1.1, 2.4];
shifted_phases = mod(phases + pi, 2*pi);

[mrx, masy, snr] = evaluate_tuned_excitation_fixture(phases, neff, sp, pp);
[mrx_shifted, masy_shifted, snr_shifted] = evaluate_tuned_excitation_fixture( ...
    shifted_phases, neff, sp, pp);

table = [
    sp.del_w(:), ...
    real(masy(:)), imag(masy(:)), ...
    real(mrx(:)), imag(mrx(:)), ...
    real(masy_shifted(:)), imag(masy_shifted(:)), ...
    real(mrx_shifted(:)), imag(mrx_shifted(:)), ...
    snr * ones(sp.numpts, 1), ...
    snr_shifted * ones(sp.numpts, 1)
];
dlmwrite( ...
    fullfile(fixture_dir, 'optimization_tuned_excitation_phase_shift.csv'), ...
    table, ...
    'precision', '%.17g');

disp(['Wrote optimization fixtures to ', fixture_dir]);

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

function matlab_code = resolve_matlab_code(repo_root)
    parent = fileparts(repo_root);
    candidates = {
        fullfile(parent, 'SpinDynamicsUpdated', 'Version_2', 'code'), ...
        fullfile(parent, 'MATLABSpinDynamics', 'SpinDynamicsUpdated', 'Version_2', 'code')
    };
    for idx = 1:numel(candidates)
        if exist(candidates{idx}, 'dir') == 7
            matlab_code = candidates{idx};
            return;
        end
    end
    matlab_code = candidates{end};
end
