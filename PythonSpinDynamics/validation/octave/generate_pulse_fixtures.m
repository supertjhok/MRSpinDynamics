% GENERATE_PULSE_FIXTURES
% Generate compact MATLAB reference CSV files for pulse helper validation.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
matlab_code = fullfile(fileparts(repo_root), 'SpinDynamicsUpdated', 'Version_2', 'code');
fixture_dir = fullfile(repo_root, 'validation', 'fixtures');

if exist(fixture_dir, 'dir') ~= 7
    mkdir(fixture_dir);
end

addpath(fullfile(matlab_code, 'Params'));
addpath(fullfile(matlab_code, 'Pulse Shape'));
addpath(fullfile(matlab_code, 'opt_pulse'));
addpath(fullfile(matlab_code, 'circuit_simulation', 'tuned_probe'));
addpath(fullfile(matlab_code, 'circuit_simulation', 'untuned_probe'));
addpath(fullfile(matlab_code, 'circuit_simulation', 'matched_probe'));

VBB = 62.5;

[sp_tuned, pp_tuned] = set_params_tuned_JMR;
sp_tuned.numpts = 17;
sp_tuned.del_w = linspace(-sp_tuned.maxoffs, sp_tuned.maxoffs, sp_tuned.numpts);
[tvect2, Icr2, tvect, Ic] = tuned_probe_lp(sp_tuned, pp_tuned);
tf = tuned_probe_rx_tf(sp_tuned, pp_tuned);
write_complex_samples( ...
    fullfile(fixture_dir, 'pulse_tuned_rectangular.csv'), ...
    tvect2, VBB * Icr2, tvect, VBB * Ic, tf);

[sp_untuned, pp_untuned] = set_params_untuned_JMR;
sp_untuned.numpts = 17;
sp_untuned.del_w = linspace(-sp_untuned.maxoffs, sp_untuned.maxoffs, sp_untuned.numpts);
[tvect2, Icr2, tvect, Ic] = untuned_probe_lp(sp_untuned, pp_untuned);
macq = ones(1, length(sp_untuned.del_w));
[~, ~, tf] = untuned_probe_rx(sp_untuned, pp_untuned, macq);
write_complex_samples( ...
    fullfile(fixture_dir, 'pulse_untuned_rectangular.csv'), ...
    tvect2, VBB * Icr2, tvect, VBB * Ic, tf);

[sp_matched, pp_matched] = set_params_matched_JMR;
sp_matched.numpts = 17;
sp_matched.del_w = linspace(-sp_matched.maxoffs, sp_matched.maxoffs, sp_matched.numpts);
sp_matched.plt_tx = 0;
sp_matched.plt_rx = 0;
[C1, C2] = matching_network_design2(sp_matched.L, sp_matched.Q, sp_matched.f0, sp_matched.Rs, 0);
sp_matched.C1 = C1;
sp_matched.C2 = C2;
pp_matched.tp = pp_matched.tref;
pp_matched.phi = pp_matched.pref;
pp_matched.amp = pp_matched.aref;
[tvec2, yr2, ~, ~, tf1, tf2] = find_coil_current_orig(sp_matched, pp_matched);
write_matched_samples( ...
    fullfile(fixture_dir, 'pulse_matched_rectangular.csv'), ...
    tvec2, yr2, tf1, tf2, C1, C2);

phi = [-0.2, 0.1, pi/5, pi/2, pi, 1.8*pi, 2*pi + 0.2];
pp_tuned.NumPhases = 8;
phiq = quantize_phase(phi, sp_tuned, pp_tuned);
dlmwrite( ...
    fullfile(fixture_dir, 'pulse_quantize_phase.csv'), ...
    [phi(:), phiq(:)], ...
    'precision', '%.17g');

segment_lengths = [4, 3, 5, 2] * 1e-6;
phases = [0.1, 1.2, 2.7, 4.0];
[adjusted, pvec, del_phi, tclk, theta] = adjust_untuned_segments_for_fixture( ...
    segment_lengths, phases, sp_untuned, pp_untuned, 8);
dlmwrite( ...
    fullfile(fixture_dir, 'pulse_untuned_segment_adjust.csv'), ...
    [segment_lengths(:), phases(:), adjusted(:), pvec(:)], ...
    'precision', '%.17g');
dlmwrite( ...
    fullfile(fixture_dir, 'pulse_untuned_segment_adjust_meta.csv'), ...
    [del_phi, tclk, theta], ...
    'precision', '%.17g');

disp(['Wrote pulse fixtures to ', fixture_dir]);

function write_complex_samples(path, t_rot, i_rot, t_raw, i_raw, tf)
    rot_idx = unique(round(linspace(1, length(t_rot), min(8, length(t_rot)))));
    raw_idx = unique(round(linspace(1, length(t_raw), min(8, length(t_raw)))));
    tf_idx = unique(round(linspace(1, length(tf), min(8, length(tf)))));
    n = max([length(rot_idx), length(raw_idx), length(tf_idx)]);
    table = nan(n, 9);
    for ii = 1:length(rot_idx)
        idx = rot_idx(ii);
        table(ii, 1:3) = [t_rot(idx), real(i_rot(idx)), imag(i_rot(idx))];
    end
    for ii = 1:length(raw_idx)
        idx = raw_idx(ii);
        table(ii, 4:6) = [t_raw(idx), real(i_raw(idx)), imag(i_raw(idx))];
    end
    for ii = 1:length(tf_idx)
        idx = tf_idx(ii);
        table(ii, 7:9) = [idx, real(tf(idx)), imag(tf(idx))];
    end
    dlmwrite(path, table, 'precision', '%.17g');
end

function write_matched_samples(path, t_rot, i_rot, tf1, tf2, C1, C2)
    rot_idx = unique(round(linspace(1, length(t_rot), min(8, length(t_rot)))));
    tf_idx = unique(round(linspace(1, length(tf1), min(8, length(tf1)))));
    n = max(length(rot_idx), length(tf_idx));
    table = nan(n, 10);
    table(1, 9:10) = [C1, C2];
    for ii = 1:length(rot_idx)
        idx = rot_idx(ii);
        table(ii, 1:3) = [t_rot(idx), real(i_rot(idx)), imag(i_rot(idx))];
    end
    for ii = 1:length(tf_idx)
        idx = tf_idx(ii);
        table(ii, 4:8) = [idx, real(tf1(idx)), imag(tf1(idx)), real(tf2(idx)), imag(tf2(idx))];
    end
    dlmwrite(path, table, 'precision', '%.17g');
end

function [tvec_adj, pvec, del_phi, tclk, theta] = adjust_untuned_segments_for_fixture(tvec, phases, sp, pp, num_phases)
    pp.NumPhases = num_phases;
    pvec = quantize_phase(phases, sp, pp);
    len = length(phases);
    tvec_adj = tvec;
    w = pp.w;
    tclk = 2*pi/(w*pp.N);
    tau = sp.L/(sp.R + pp.Rsref(2));
    theta = -atan2(w*tau, 1);
    phi_1 = pi/2 - theta;
    del_phi = phi_1 - pvec(1);
    pvec = pvec + del_phi;
    for ii = 1:len-1
        alpha = mod(-(pvec(ii) + pvec(ii+1))/2 - theta, pi);
        if alpha <= pi/2
            tadj = alpha/w;
        else
            tadj = -(pi - alpha)/w;
        end
        tadj = round(tadj/tclk)*tclk;
        tvec_adj(ii) = tvec_adj(ii) + tadj;
        tvec_adj(ii+1) = tvec_adj(ii+1) - tadj;
    end
    tmp = zeros(1, 2);
    tmp(1) = pi/2 - pvec(end) - theta;
    tmp(2) = 3*pi/2 - pvec(end) - theta;
    [~, ind] = min(abs(tmp));
    tadj = tmp(ind)/w;
    tadj = round(tadj/tclk)*tclk;
    tvec_adj(end) = tvec_adj(end) + tadj;
end
