% GENERATE_BASIC_FIXTURES
% Generate Octave/MATLAB reference CSV files for early Python validation.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
matlab_code = fullfile(fileparts(repo_root), 'SpinDynamicsUpdated', 'Version_2', 'code');
fixture_dir = fullfile(repo_root, 'validation', 'fixtures');

if exist(fixture_dir, 'dir') ~= 7
    mkdir(fixture_dir);
end

addpath(fullfile(matlab_code, 'calc_echo'));
addpath(fullfile(matlab_code, 'calc_masy'));
addpath(fullfile(matlab_code, 'calc_rot'));
addpath(fullfile(matlab_code, 'Params'));
addpath(fullfile(matlab_code, 'sim_spin_dynamics_arb'));
addpath(fullfile(matlab_code, 'sim_spin_dynamics_asymp'));

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

disp(['Wrote fixtures to ', fixture_dir]);
