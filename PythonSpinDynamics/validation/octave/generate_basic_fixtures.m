% GENERATE_BASIC_FIXTURES
% Generate Octave/MATLAB reference CSV files for early Python validation.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
matlab_code = fullfile(fileparts(repo_root), 'SpinDynamicsUpdated', 'Version_2', 'code');
fixture_dir = fullfile(repo_root, 'validation', 'fixtures');

if exist(fixture_dir, 'dir') ~= 7
    mkdir(fixture_dir);
end

addpath(fullfile(matlab_code, 'calc_echo'));
addpath(fullfile(matlab_code, 'calc_FID_decay'));
addpath(fullfile(matlab_code, 'calc_macq'));
addpath(fullfile(matlab_code, 'calc_masy'));
addpath(fullfile(matlab_code, 'calc_rot'));
addpath(fullfile(matlab_code, 'Params'));
addpath(fullfile(matlab_code, 'Sim_FID'));
addpath(fullfile(matlab_code, 'sim_spin_dynamics_arb'));
addpath(fullfile(matlab_code, 'sim_spin_dynamics_asymp'));
addpath(fullfile(matlab_code, 'circuit_simulation', 'tuned_probe'));
addpath(fullfile(matlab_code, 'circuit_simulation', 'untuned_probe'));
addpath(fullfile(matlab_code, 'circuit_simulation', 'matched_probe'));

del_w = linspace(-4, 4, 17);
spect = exp(-0.25 * del_w.^2) .* exp(1i * 0.2 * del_w);
[echo_ref, tvect_ref] = calc_time_domain_echo(spect, del_w, 0, 0);

echo_table = [real(echo_ref(:)), imag(echo_ref(:)), tvect_ref(:)];
dlmwrite(fullfile(fixture_dir, 'calc_time_domain_echo.csv'), echo_table, 'precision', '%.17g');

mrx_arb = exp(-0.2 * del_w.^2) .* exp(1i * (0.3 * del_w + 0.05 * del_w.^2));
tacq_arb = 4 * pi;
tdw_arb = tacq_arb / 32;
[echo_arb_ref, tvect_arb_ref] = calc_time_domain_echo_arb(mrx_arb, del_w, tacq_arb, tdw_arb, 0);
echo_arb_table = [real(echo_arb_ref(:)), imag(echo_arb_ref(:)), tvect_arb_ref(:)];
dlmwrite(fullfile(fixture_dir, 'calc_time_domain_echo_arb.csv'), echo_arb_table, 'precision', '%.17g');

tp = [pi/2, 0.5, pi/3];
phi = [pi/2, 0, pi/4];
amp = [1, 0, 0.75];
t_acq = 2 * pi;
neff = zeros(3, length(del_w));
neff(1, :) = cos(0.15 * del_w);
neff(2, :) = sin(0.15 * del_w);
neff(3, :) = 0.25;
neff = neff ./ sqrt(sum(neff.^2, 1));

masy_ref = sim_spin_dynamics_asymp_mag3(tp, phi, amp, neff, del_w, t_acq);
asymp_table = [real(masy_ref(:)), imag(masy_ref(:)), del_w(:)];
dlmwrite(fullfile(fixture_dir, 'sim_spin_dynamics_asymp_mag3.csv'), asymp_table, 'precision', '%.17g');

tp_rot = [0.8, pi, 0.4, pi/3];
phi_rot = [0, pi/2, 0, pi/5];
amp_rot = [0, 1, 0, 0.6];
del_w_rot = linspace(-3, 3, 13);
[n4_ref, alpha_ref] = calc_rot_axis_arba4(tp_rot, phi_rot, amp_rot, del_w_rot, 0);
n3_ref = calc_rot_axis_arba3(tp_rot, phi_rot, amp_rot, del_w_rot, 0);
rot_table = [del_w_rot(:), n3_ref.', n4_ref.', alpha_ref(:)];
dlmwrite(fullfile(fixture_dir, 'calc_rot_axis_arba.csv'), rot_table, 'precision', '%.17g');

sp = struct();
sp.del_w = linspace(-5, 5, 21);
sp.plt_axis = 0;
sp.plt_rx = 0;

pp = struct();
pp.T_90 = 25e-6;
pp.tref = [75e-6, 50e-6, 75e-6];
pp.pref = [0, 0, 0];
pp.aref = [0, 1, 0];
pp.texc = [25e-6];
pp.pexc = [pi/2];
pp.aexc = [1];
pp.tacq = [150e-6];

masy_ideal_ref = calc_masy_ideal(sp, pp);
masy_ideal_table = [real(masy_ideal_ref(:)), imag(masy_ideal_ref(:)), sp.del_w(:)];
dlmwrite(fullfile(fixture_dir, 'calc_masy_ideal.csv'), masy_ideal_table, 'precision', '%.17g');

[sp_default, pp_default] = set_params_ideal();
params_table = [
    sp_default.k;
    sp_default.T;
    sp_default.gamma;
    sp_default.grad;
    sp_default.D;
    sp_default.f0;
    sp_default.fin;
    sp_default.m0;
    sp_default.mth;
    sp_default.numpts;
    sp_default.maxoffs;
    sp_default.del_w(1);
    sp_default.del_w(end);
    sp_default.mf_type;
    sp_default.plt_tx;
    sp_default.plt_rx;
    sp_default.plt_sequence;
    sp_default.plt_axis;
    sp_default.plt_mn;
    sp_default.plt_echo;
    pp_default.N;
    pp_default.T_90;
    pp_default.T_180;
    pp_default.psi;
    pp_default.preDelay;
    pp_default.postDelay;
    pp_default.texc(1);
    pp_default.pexc(1);
    pp_default.aexc(1);
    pp_default.tcorr;
    pp_default.tref(:);
    pp_default.pref(:);
    pp_default.aref(:);
    pp_default.pcycle;
    pp_default.tacq(1);
    pp_default.tdw;
    pp_default.amp_zero
];
dlmwrite(fullfile(fixture_dir, 'set_params_ideal.csv'), params_table, 'precision', '%.17g');

[sp_fid_default, pp_fid_default] = set_params_ideal_FID();
params_fid_table = [
    sp_fid_default.k;
    sp_fid_default.T;
    sp_fid_default.f0;
    sp_fid_default.fin;
    sp_fid_default.m0;
    sp_fid_default.mth;
    sp_fid_default.numpts;
    sp_fid_default.maxoffs;
    sp_fid_default.del_w(1);
    sp_fid_default.del_w(end);
    sp_fid_default.w_1(1);
    sp_fid_default.w_1r(1);
    sp_fid_default.T1(1);
    sp_fid_default.T2(1);
    sp_fid_default.mf_type;
    sp_fid_default.plt_tx;
    sp_fid_default.plt_rx;
    sp_fid_default.plt_sequence;
    sp_fid_default.plt_axis;
    sp_fid_default.plt_mn;
    sp_fid_default.plt_echo;
    pp_fid_default.N;
    pp_fid_default.T_90;
    pp_fid_default.acqDelay;
    pp_fid_default.acqTpTime;
    pp_fid_default.psi;
    pp_fid_default.tacq;
    pp_fid_default.tdw;
    pp_fid_default.amp_zero
];
dlmwrite(fullfile(fixture_dir, 'set_params_ideal_fid.csv'), params_fid_table, 'precision', '%.17g');

sp_fid = sp_fid_default;
sp_fid.numpts = 23;
sp_fid.del_w = linspace(-6, 6, sp_fid.numpts);
sp_fid.w_1 = 0.9 + 0.1 * cos(sp_fid.del_w / 2);
sp_fid.w_1r = ones(1, sp_fid.numpts);
sp_fid.T1 = (1.5 + 0.1 * cos(sp_fid.del_w / 3));
sp_fid.T2 = (1.2 + 0.1 * sin(sp_fid.del_w / 4));
sp_fid.plt_sequence = 0;
sp_fid.plt_echo = 0;
pp_fid = pp_fid_default;

params_fid = struct();
params_fid.tp = [pp_fid.T_90, pp_fid.acqDelay, pp_fid.tacq, pp_fid.acqDelay];
params_fid.phi = [0, 0, 0, 0];
params_fid.amp = [1, 0, 0, 0];
params_fid.acq = [0, 0, 1, 0];
params_fid.grad = [0, 0, 0, 0];
params_fid.len_acq = pp_fid.tacq;
params_fid.del_w = sp_fid.del_w;
params_fid.w_1 = sp_fid.w_1;
params_fid.m0 = sp_fid.m0;
params_fid.T1n = sp_fid.T1;
params_fid.T2n = sp_fid.T2;
params_fid.mth = sp_fid.mth;

[macq_fid_ref, tacq_fid_ref] = calc_macq_fid(sp_fid, pp_fid, params_fid);
fid_macq_table = [real(macq_fid_ref(:)), imag(macq_fid_ref(:)), sp_fid.del_w(:)];
dlmwrite(fullfile(fixture_dir, 'sim_fid_ideal_macq.csv'), fid_macq_table, 'precision', '%.17g');

tdw_fid_ref = (pi / 2) * pp_fid.tdw / pp_fid.T_90;
[echo_fid_ref, tvect_fid_ref] = calc_FID_time_domain( ...
    macq_fid_ref, sp_fid.del_w, tacq_fid_ref, tdw_fid_ref, 0);
fid_echo_table = [real(echo_fid_ref(:)), imag(echo_fid_ref(:)), tvect_fid_ref(:)];
dlmwrite(fullfile(fixture_dir, 'sim_fid_ideal_echo.csv'), fid_echo_table, 'precision', '%.17g');

[params_tuned_default, sp_tuned_default, pp_tuned_default] = set_params_tuned_Orig();
tuned_params_table = [
    sp_tuned_default.k;
    sp_tuned_default.T;
    sp_tuned_default.gamma;
    sp_tuned_default.f0;
    sp_tuned_default.fin;
    sp_tuned_default.w0;
    sp_tuned_default.L;
    sp_tuned_default.Q;
    sp_tuned_default.R;
    sp_tuned_default.C;
    sp_tuned_default.Rs;
    sp_tuned_default.Vs;
    sp_tuned_default.Rin;
    sp_tuned_default.Cin;
    sp_tuned_default.Rd;
    sp_tuned_default.NF;
    sp_tuned_default.vn;
    sp_tuned_default.in;
    sp_tuned_default.m0;
    sp_tuned_default.mth;
    sp_tuned_default.numpts;
    sp_tuned_default.maxoffs;
    sp_tuned_default.del_w(1);
    sp_tuned_default.del_w(end);
    sp_tuned_default.mf_type;
    sp_tuned_default.plt_tx;
    sp_tuned_default.plt_rx;
    sp_tuned_default.plt_sequence;
    sp_tuned_default.plt_axis;
    sp_tuned_default.plt_mn;
    sp_tuned_default.plt_echo;
    sp_tuned_default.sens;
    pp_tuned_default.w;
    pp_tuned_default.N;
    pp_tuned_default.T_90;
    pp_tuned_default.T_180;
    pp_tuned_default.psi;
    pp_tuned_default.preDelay;
    pp_tuned_default.postDelay;
    pp_tuned_default.texc(1);
    pp_tuned_default.pexc(1);
    pp_tuned_default.aexc(1);
    pp_tuned_default.tcorr;
    pp_tuned_default.tqs;
    pp_tuned_default.trd;
    pp_tuned_default.tref(:);
    pp_tuned_default.pref(:);
    pp_tuned_default.aref(:);
    pp_tuned_default.Rsref(:);
    pp_tuned_default.pcycle;
    pp_tuned_default.tacq(1);
    pp_tuned_default.tdw;
    pp_tuned_default.amp_zero;
    params_tuned_default.texc(1);
    params_tuned_default.pexc(1);
    params_tuned_default.aexc(1);
    params_tuned_default.trd;
    params_tuned_default.tref(1);
    params_tuned_default.pref(1);
    params_tuned_default.aref(1);
    params_tuned_default.tfp;
    params_tuned_default.tqs;
    params_tuned_default.tacq(1);
    params_tuned_default.Rs(:);
    params_tuned_default.pcycle
];
dlmwrite(fullfile(fixture_dir, 'set_params_tuned_orig.csv'), tuned_params_table, 'precision', '%.17g');

sp_tuned = sp_tuned_default;
sp_tuned.numpts = 21;
sp_tuned.del_w = linspace(-5, 5, sp_tuned.numpts);
sp_tuned.plt_tx = 0;
sp_tuned.plt_rx = 0;
sp_tuned.plt_echo = 0;
pp_tuned = pp_tuned_default;
params_tuned = params_tuned_default;

[tvect_tuned, icr_tuned, ~, ~] = tuned_probe_lp_Orig(sp_tuned, pp_tuned);
tuned_lp_table = [
    tvect_tuned(:), real(icr_tuned(:)), imag(icr_tuned(:))
];
dlmwrite(fullfile(fixture_dir, 'tuned_probe_lp_orig.csv'), tuned_lp_table, 'precision', '%.17g');

[mrx_tuned_ref, masy_tuned_ref, snr_tuned_ref] = calc_masy_tuned_probe_lp_Orig( ...
    params_tuned, sp_tuned, pp_tuned);
tuned_masy_table = [
    sp_tuned.del_w(:), ...
    real(masy_tuned_ref(:)), imag(masy_tuned_ref(:)), ...
    real(mrx_tuned_ref(:)), imag(mrx_tuned_ref(:)), ...
    snr_tuned_ref * ones(sp_tuned.numpts, 1)
];
dlmwrite(fullfile(fixture_dir, 'calc_masy_tuned_probe_lp_orig.csv'), tuned_masy_table, 'precision', '%.17g');

[params_untuned_default, sp_untuned_default, pp_untuned_default] = set_params_untuned_Orig();
untuned_params_table = [
    sp_untuned_default.k;
    sp_untuned_default.T;
    sp_untuned_default.gamma;
    sp_untuned_default.f0;
    sp_untuned_default.fin;
    sp_untuned_default.w0;
    sp_untuned_default.L;
    sp_untuned_default.Q;
    sp_untuned_default.R;
    sp_untuned_default.C;
    sp_untuned_default.Rs;
    sp_untuned_default.Vs;
    sp_untuned_default.Rin;
    sp_untuned_default.Cin;
    sp_untuned_default.Rd;
    sp_untuned_default.Rdup;
    sp_untuned_default.Nrx;
    sp_untuned_default.krx;
    sp_untuned_default.L1;
    sp_untuned_default.R1;
    sp_untuned_default.L2;
    sp_untuned_default.R2;
    sp_untuned_default.NF;
    sp_untuned_default.vn;
    sp_untuned_default.in;
    sp_untuned_default.m0;
    sp_untuned_default.mth;
    sp_untuned_default.numpts;
    sp_untuned_default.maxoffs;
    sp_untuned_default.del_w(1);
    sp_untuned_default.del_w(end);
    sp_untuned_default.mf_type;
    sp_untuned_default.plt_tx;
    sp_untuned_default.plt_rx;
    sp_untuned_default.plt_sequence;
    sp_untuned_default.plt_axis;
    sp_untuned_default.plt_mn;
    sp_untuned_default.plt_echo;
    sp_untuned_default.sens;
    pp_untuned_default.w;
    pp_untuned_default.N;
    pp_untuned_default.T_90;
    pp_untuned_default.T_180;
    pp_untuned_default.psi;
    pp_untuned_default.preDelay;
    pp_untuned_default.postDelay;
    pp_untuned_default.texc(1);
    pp_untuned_default.pexc(1);
    pp_untuned_default.aexc(1);
    pp_untuned_default.tcorr;
    pp_untuned_default.tqs;
    pp_untuned_default.trd;
    pp_untuned_default.tref(:);
    pp_untuned_default.pref(:);
    pp_untuned_default.aref(:);
    pp_untuned_default.Rsref(:);
    pp_untuned_default.tacq(1);
    pp_untuned_default.tdw;
    pp_untuned_default.amp_zero;
    params_untuned_default.texc(1);
    params_untuned_default.pexc(1);
    params_untuned_default.aexc(1);
    params_untuned_default.trd;
    params_untuned_default.tref(1);
    params_untuned_default.pref(1);
    params_untuned_default.aref(1);
    params_untuned_default.tfp;
    params_untuned_default.tqs;
    params_untuned_default.tacq(1);
    params_untuned_default.Rs(:);
    params_untuned_default.pcycle
];
dlmwrite(fullfile(fixture_dir, 'set_params_untuned_orig.csv'), untuned_params_table, 'precision', '%.17g');

sp_untuned = sp_untuned_default;
sp_untuned.numpts = 21;
sp_untuned.del_w = linspace(-5, 5, sp_untuned.numpts);
sp_untuned.plt_tx = 0;
sp_untuned.plt_rx = 0;
sp_untuned.plt_echo = 0;
sp_untuned.plt_axis = 0;
pp_untuned = pp_untuned_default;
params_untuned = params_untuned_default;

[tvect_untuned, icr_untuned, ~, ~] = untuned_probe_lp(sp_untuned, pp_untuned);
untuned_lp_table = [
    tvect_untuned(:), real(icr_untuned(:)), imag(icr_untuned(:))
];
dlmwrite(fullfile(fixture_dir, 'untuned_probe_lp.csv'), untuned_lp_table, 'precision', '%.17g');

[mrx_untuned_ref, masy_untuned_ref, snr_untuned_ref] = calc_masy_untuned_probe_lp( ...
    params_untuned, sp_untuned, pp_untuned);
untuned_masy_table = [
    sp_untuned.del_w(:), ...
    real(masy_untuned_ref(:)), imag(masy_untuned_ref(:)), ...
    real(mrx_untuned_ref(:)), imag(mrx_untuned_ref(:)), ...
    snr_untuned_ref * ones(sp_untuned.numpts, 1)
];
dlmwrite(fullfile(fixture_dir, 'calc_masy_untuned_probe_lp.csv'), untuned_masy_table, 'precision', '%.17g');

try
    [sp_matched_default, pp_matched_default] = set_params_matched_Orig();
    matched_params_table = [
        sp_matched_default.k;
        sp_matched_default.T;
        sp_matched_default.gamma;
        sp_matched_default.grad;
        sp_matched_default.D;
        sp_matched_default.f0;
        sp_matched_default.fin;
        sp_matched_default.L;
        sp_matched_default.Q;
        sp_matched_default.R;
        sp_matched_default.Rs;
        sp_matched_default.Rin;
        sp_matched_default.NF;
        sp_matched_default.m0;
        sp_matched_default.mth;
        sp_matched_default.numpts;
        sp_matched_default.maxoffs;
        sp_matched_default.del_w(1);
        sp_matched_default.del_w(end);
        sp_matched_default.mf_type;
        sp_matched_default.plt_tx;
        sp_matched_default.plt_rx;
        sp_matched_default.plt_sequence;
        sp_matched_default.plt_axis;
        sp_matched_default.plt_mn;
        sp_matched_default.plt_echo;
        pp_matched_default.N;
        pp_matched_default.T_90;
        pp_matched_default.T_180;
        pp_matched_default.psi;
        pp_matched_default.preDelay;
        pp_matched_default.postDelay;
        pp_matched_default.texc(1);
        pp_matched_default.pexc(1);
        pp_matched_default.aexc(1);
        pp_matched_default.tcorr;
        pp_matched_default.trd;
        pp_matched_default.tref(:);
        pp_matched_default.pref(:);
        pp_matched_default.aref(:);
        pp_matched_default.tacq(1);
        pp_matched_default.tdw;
        pp_matched_default.amp_zero
    ];
    dlmwrite(fullfile(fixture_dir, 'set_params_matched_orig.csv'), matched_params_table, 'precision', '%.17g');

    sp_matched = sp_matched_default;
    sp_matched.numpts = 11;
    sp_matched.del_w = linspace(-4, 4, sp_matched.numpts);
    sp_matched.plt_tx = 0;
    sp_matched.plt_rx = 0;
    sp_matched.plt_echo = 0;
    sp_matched.plt_axis = 0;
    sp_matched.plt_mn = 0;
    pp_matched = pp_matched_default;
    [c1_matched, c2_matched] = matching_network_design2( ...
        sp_matched.L, sp_matched.Q, sp_matched.f0, sp_matched.Rs, 0);
    sp_matched.C1 = c1_matched;
    sp_matched.C2 = c2_matched;

    pp_matched_current = pp_matched;
    pp_matched_current.tp = [pp_matched.texc pp_matched.trd];
    pp_matched_current.phi = [pp_matched.pexc 0];
    pp_matched_current.amp = [pp_matched.aexc 0];
    [tvect_matched, icr_matched, tf1_matched, tf2_matched] = find_coil_current( ...
        sp_matched, pp_matched_current);
    matched_current_table = [
        tvect_matched(:), real(icr_matched(:)), imag(icr_matched(:))
    ];
    dlmwrite(fullfile(fixture_dir, 'find_coil_current_matched.csv'), matched_current_table, 'precision', '%.17g');

    [mrx_matched_ref, masy_matched_ref, snr_matched_ref] = calc_masy_matched_probe_Orig( ...
        sp_matched, pp_matched);
    matched_masy_table = [
        sp_matched.del_w(:), ...
        real(masy_matched_ref(:)), imag(masy_matched_ref(:)), ...
        real(mrx_matched_ref(:)), imag(mrx_matched_ref(:)), ...
        snr_matched_ref * ones(sp_matched.numpts, 1), ...
        c1_matched * ones(sp_matched.numpts, 1), ...
        c2_matched * ones(sp_matched.numpts, 1)
    ];
    dlmwrite(fullfile(fixture_dir, 'calc_masy_matched_probe_orig.csv'), matched_masy_table, 'precision', '%.17g');
catch matched_fixture_error
    disp(['Skipping matched-probe fixtures: ', matched_fixture_error.message]);
end

sp_rot = struct();
sp_rot.del_w = linspace(-2.5, 2.5, 11);
sp_rot.w_1 = 0.85 + 0.05 * cos(sp_rot.del_w);
pp_rot = struct();
pp_rot.tp = [0.25, 0.5, 0.75];
pp_rot.phi = [0, pi/3, -pi/5];
pp_rot.amp = [1.0, 0.6, 1.2];
Rtot_ref = calc_rotation_matrix(sp_rot, pp_rot);
rotmat_table = [
    sp_rot.del_w(:), real(Rtot_ref.R_00(:)), imag(Rtot_ref.R_00(:)), ...
    real(Rtot_ref.R_0p(:)), imag(Rtot_ref.R_0p(:)), ...
    real(Rtot_ref.R_0m(:)), imag(Rtot_ref.R_0m(:)), ...
    real(Rtot_ref.R_p0(:)), imag(Rtot_ref.R_p0(:)), ...
    real(Rtot_ref.R_m0(:)), imag(Rtot_ref.R_m0(:)), ...
    real(Rtot_ref.R_pp(:)), imag(Rtot_ref.R_pp(:)), ...
    real(Rtot_ref.R_mm(:)), imag(Rtot_ref.R_mm(:)), ...
    real(Rtot_ref.R_pm(:)), imag(Rtot_ref.R_pm(:)), ...
    real(Rtot_ref.R_mp(:)), imag(Rtot_ref.R_mp(:)), ...
    sp_rot.w_1(:)
];
dlmwrite(fullfile(fixture_dir, 'calc_rotation_matrix.csv'), rotmat_table, 'precision', '%.17g');

numpts_arb = 19;
del_w_arb = linspace(-6, 6, numpts_arb);
sp_arb = struct();
sp_arb.del_w = del_w_arb;
sp_arb.w_1 = 1 + 0.08 * sin(del_w_arb / 2);

Rtot_arb = cell(1, 2);
pp_pulse = struct();
pp_pulse.tp = pi / 2;
pp_pulse.phi = pi / 2;
pp_pulse.amp = 1;
Rtot_arb{1} = calc_rotation_matrix(sp_arb, pp_pulse);
pp_pulse.tp = [0.4 * pi, 0.6 * pi];
pp_pulse.phi = [0, pi / 4];
pp_pulse.amp = [0.8, 1.1];
Rtot_arb{2} = calc_rotation_matrix(sp_arb, pp_pulse);

params_arb = struct();
params_arb.tp = [pi/2, 1.2*pi, 0.7*pi, 0.9*pi, 0.5*pi, 1.1*pi];
params_arb.pul = [1, 0, 2, 0, 0, 2];
params_arb.Rtot = Rtot_arb;
params_arb.amp = [1, 0, 1, 0, 0, 1];
params_arb.acq = [0, 1, 0, 1, 1, 0];
params_arb.grad = [0, 0.2, 0, -0.15, 0.1, 0];
params_arb.del_w = del_w_arb;
params_arb.del_wg = linspace(-1, 1, numpts_arb);
params_arb.T1n = 120 + 10 * cos(del_w_arb / 3);
params_arb.T2n = 45 + 5 * sin(del_w_arb / 4);
params_arb.m0 = 0.9 + 0.05 * cos(del_w_arb);
params_arb.mth = 1.1 + 0.03 * sin(del_w_arb);

macq_arb_ref = sim_spin_dynamics_arb10(params_arb);
arb_table = zeros(numel(macq_arb_ref), 4);
row = 1;
for ii = 1:size(macq_arb_ref, 1)
    for jj = 1:size(macq_arb_ref, 2)
        arb_table(row, :) = [ii, jj, real(macq_arb_ref(ii, jj)), imag(macq_arb_ref(ii, jj))];
        row = row + 1;
    end
end
dlmwrite(fullfile(fixture_dir, 'sim_spin_dynamics_arb10.csv'), arb_table, 'precision', '%.17g');

sp_acq = struct();
sp_acq.del_w = linspace(-5, 5, 17);
sp_acq.del_wg = linspace(-0.75, 0.75, 17);
sp_acq.w_1 = 0.95 + 0.07 * cos(sp_acq.del_w / 2);
sp_acq.T1 = 1.4 + 0.08 * cos(sp_acq.del_w / 3);
sp_acq.T2 = 0.9 + 0.05 * sin(sp_acq.del_w / 4);
sp_acq.m0 = 0.8 + 0.04 * cos(sp_acq.del_w);
sp_acq.mth = 1.05 + 0.02 * sin(sp_acq.del_w);
sp_acq.plt_sequence = 0;

pp_acq = struct();
pp_acq.T_90 = 25e-6;
pp_acq.tacq = 150e-6;
pp_acq.tp = [pi/2, 0.35*pi, 0.9*pi, 0.4*pi, 0.55*pi, 1.1*pi];
pp_acq.amp = [1, 0, 1, 0, 0, 1];
pp_acq.acq = [0, 1, 0, 1, 1, 0];
pp_acq.grad = [0, 0.25, 0, -0.2, 0.15, 0];
pp_acq.pul = [1, 0, 2, 0, 0, 3];
pp_acq.Rtot = cell(1, 3);
pp_rot_acq = struct();
pp_rot_acq.tp = pi/2;
pp_rot_acq.phi = pi/2;
pp_rot_acq.amp = 1;
pp_acq.Rtot{1} = calc_rotation_matrix(sp_acq, pp_rot_acq);
pp_rot_acq.tp = [0.45*pi, 0.45*pi];
pp_rot_acq.phi = [0, pi/4];
pp_rot_acq.amp = [0.8, 1.05];
pp_acq.Rtot{2} = calc_rotation_matrix(sp_acq, pp_rot_acq);
pp_rot_acq.tp = 1.1*pi;
pp_rot_acq.phi = -pi/5;
pp_rot_acq.amp = 0.9;
pp_acq.Rtot{3} = calc_rotation_matrix(sp_acq, pp_rot_acq);

macq_acq_ref = calc_macq_ideal_probe_relax4(sp_acq, pp_acq);
acq_table = zeros(numel(macq_acq_ref), 4);
row = 1;
for ii = 1:size(macq_acq_ref, 1)
    for jj = 1:size(macq_acq_ref, 2)
        acq_table(row, :) = [ii, jj, real(macq_acq_ref(ii, jj)), imag(macq_acq_ref(ii, jj))];
        row = row + 1;
    end
end
dlmwrite(fullfile(fixture_dir, 'calc_macq_ideal_probe_relax4.csv'), acq_table, 'precision', '%.17g');

sp_tuned_acq = sp_acq;
sp_tuned_acq.tf = (0.8 + 0.03 * sp_acq.del_w) .* exp(1i * 0.15 * sp_acq.del_w);
sp_tuned_acq.w_1r = 0.9 + 0.02 * cos(sp_acq.del_w);
[macq_tuned_acq_ref, mrx_tuned_acq_ref] = calc_macq_tuned_probe_relax4( ...
    sp_tuned_acq, pp_acq);
tuned_acq_table = zeros(numel(macq_tuned_acq_ref), 6);
row = 1;
for ii = 1:size(macq_tuned_acq_ref, 1)
    for jj = 1:size(macq_tuned_acq_ref, 2)
        tuned_acq_table(row, :) = [
            ii, jj, ...
            real(macq_tuned_acq_ref(ii, jj)), imag(macq_tuned_acq_ref(ii, jj)), ...
            real(mrx_tuned_acq_ref(ii, jj)), imag(mrx_tuned_acq_ref(ii, jj))
        ];
        row = row + 1;
    end
end
dlmwrite(fullfile(fixture_dir, 'calc_macq_tuned_probe_relax4.csv'), tuned_acq_table, 'precision', '%.17g');

sp_matched_acq = sp_acq;
sp_matched_acq.tf2 = (0.7 - 0.02 * sp_acq.del_w) .* exp(-1i * 0.11 * sp_acq.del_w);
sp_matched_acq.w_1r = 1.0 + 0.03 * sin(sp_acq.del_w / 2);
[macq_matched_acq_ref, mrx_matched_acq_ref] = calc_macq_matched_probe_relax4( ...
    sp_matched_acq, pp_acq);
matched_acq_table = zeros(numel(macq_matched_acq_ref), 6);
row = 1;
for ii = 1:size(macq_matched_acq_ref, 1)
    for jj = 1:size(macq_matched_acq_ref, 2)
        matched_acq_table(row, :) = [
            ii, jj, ...
            real(macq_matched_acq_ref(ii, jj)), imag(macq_matched_acq_ref(ii, jj)), ...
            real(mrx_matched_acq_ref(ii, jj)), imag(mrx_matched_acq_ref(ii, jj))
        ];
        row = row + 1;
    end
end
dlmwrite(fullfile(fixture_dir, 'calc_macq_matched_probe_relax4.csv'), matched_acq_table, 'precision', '%.17g');

numpts_train = 17;
num_echoes_train = 4;
[sp_train_default, pp_train_default] = set_params_ideal();
del_w_train = linspace(-5, 5, numpts_train);
w_1n_train = (pi / 2) / pp_train_default.T_90;

sp_train = struct();
sp_train.del_w = del_w_train;
sp_train.del_wg = zeros(1, numpts_train);
sp_train.w_1 = ones(1, numpts_train);
sp_train.T1 = 1.7 * ones(1, numpts_train);
sp_train.T2 = 1.1 * ones(1, numpts_train);
sp_train.m0 = sp_train_default.m0 * ones(1, numpts_train);
sp_train.mth = sp_train_default.mth * ones(1, numpts_train);
sp_train.plt_sequence = 0;

pp_train_in = struct();
pp_train_in.tp = w_1n_train * pp_train_default.texc;
pp_train_in.phi = pp_train_default.pexc;
pp_train_in.amp = pp_train_default.aexc;
Rtot_train = cell(1, 3);
Rtot_train{1} = calc_rotation_matrix(sp_train, pp_train_in);
pp_train_in.phi = pp_train_default.pexc + pi;
Rtot_train{2} = calc_rotation_matrix(sp_train, pp_train_in);
pp_train_in.tp = w_1n_train * pp_train_default.tref(2:end-1);
pp_train_in.phi = pp_train_default.pref(2:end-1);
pp_train_in.amp = pp_train_default.aref(2:end-1);
Rtot_train{3} = calc_rotation_matrix(sp_train, pp_train_in);

texc_train = [pi/2, w_1n_train * pp_train_default.tcorr];
aexc_train = [1, 0];
pexc1_train = [1, 0];
pexc2_train = [2, 0];
acq_exc_train = [0, 0];
grad_exc_train = [0, 0];

tref_train = repmat(w_1n_train * pp_train_default.tref, 1, num_echoes_train);
pref_train = repmat([0, 3, 0], 1, num_echoes_train);
aref_train = repmat([0, 1, 0], 1, num_echoes_train);
acq_ref_train = repmat([0, 0, 1], 1, num_echoes_train);
grad_ref_train = zeros(1, 3 * num_echoes_train);

pp_train = struct();
pp_train.T_90 = pp_train_default.T_90;
pp_train.tp = [texc_train, tref_train];
pp_train.amp = [aexc_train, aref_train];
pp_train.acq = [acq_exc_train, acq_ref_train];
pp_train.grad = [grad_exc_train, grad_ref_train];
pp_train.Rtot = Rtot_train;

pp_train.pul = [pexc1_train, pref_train];
mrx_train_1 = calc_macq_ideal_probe_relax4(sp_train, pp_train);
pp_train.pul = [pexc2_train, pref_train];
mrx_train_2 = calc_macq_ideal_probe_relax4(sp_train, pp_train);
mrx_train = (mrx_train_1 - mrx_train_2) / 2;

tacq_train = (pi / 2) * pp_train_default.tacq(1) / pp_train_default.T_90;
tdw_train = (pi / 2) * pp_train_default.tdw / pp_train_default.T_90;
nacq_train = round(tacq_train / tdw_train) + 1;
tvect_train = linspace(-tacq_train / 2, tacq_train / 2, nacq_train).';
isoc_train = exp(1i * tvect_train * del_w_train);
echo_train = (isoc_train * mrx_train.').';
echo_int_train = trapz(tvect_train, echo_train, 2);

train_mrx_table = zeros(numel(mrx_train), 4);
row = 1;
for ii = 1:size(mrx_train, 1)
    for jj = 1:size(mrx_train, 2)
        train_mrx_table(row, :) = [ii, jj, real(mrx_train(ii, jj)), imag(mrx_train(ii, jj))];
        row = row + 1;
    end
end
dlmwrite(fullfile(fixture_dir, 'run_ideal_cpmg_train_mrx.csv'), train_mrx_table, 'precision', '%.17g');

train_echo_table = zeros(numel(echo_train), 5);
row = 1;
for ii = 1:size(echo_train, 1)
    for jj = 1:size(echo_train, 2)
        train_echo_table(row, :) = [
            ii, jj, real(echo_train(ii, jj)), imag(echo_train(ii, jj)), tvect_train(jj)
        ];
        row = row + 1;
    end
end
dlmwrite(fullfile(fixture_dir, 'run_ideal_cpmg_train_echo.csv'), train_echo_table, 'precision', '%.17g');

train_int_table = [
    (1:num_echoes_train).', real(echo_int_train(:)), imag(echo_int_train(:))
];
dlmwrite(fullfile(fixture_dir, 'run_ideal_cpmg_train_integrals.csv'), train_int_table, 'precision', '%.17g');

disp(['Wrote fixtures to ', fixture_dir]);
