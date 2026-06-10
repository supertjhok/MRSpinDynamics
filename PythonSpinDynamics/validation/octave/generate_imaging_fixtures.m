% GENERATE_IMAGING_FIXTURES
% Generate compact MATLAB reference CSV files for CPMG imaging validation.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
matlab_code = fullfile(fileparts(repo_root), 'SpinDynamicsUpdated', 'Version_2', 'code');
fixture_dir = fullfile(repo_root, 'validation', 'fixtures');

if exist(fixture_dir, 'dir') ~= 7
    mkdir(fixture_dir);
end

addpath(fullfile(matlab_code, 'calc_macq'));
addpath(fullfile(matlab_code, 'calc_rot'));
addpath(fullfile(matlab_code, 'create_fields'));
addpath(fullfile(matlab_code, 'Params'));
addpath(fullfile(matlab_code, 'Sim_CPMG'));
addpath(fullfile(matlab_code, 'sim_spin_dynamics_arb'));
addpath(fullfile(matlab_code, 'circuit_simulation', 'tuned_probe'));
addpath(fullfile(matlab_code, 'circuit_simulation', 'matched_probe'));

params = struct();
params.NE = 2;
params.TE = 0.2e-3;
params.Tgrad = 0.5e-3;
params.rho = [0.0, 1.0; 1.0, 0.35];
params.T1map = 5e-3 * ones(2, 2);
params.T2map = 5e-3 * ones(2, 2);
params.pxz = [2, 2];
params.FOV = [20, 20];

write_imaging_fixture( ...
    fullfile(fixture_dir, 'run_ideal_cpmg_imaging_kspace.csv'), ...
    sim_cpmg_ideal_probe_img(params));
write_imaging_fixture( ...
    fullfile(fixture_dir, 'run_tuned_cpmg_imaging_kspace.csv'), ...
    sim_cpmg_tuned_probe_img(params));
write_imaging_fixture( ...
    fullfile(fixture_dir, 'run_matched_cpmg_imaging_kspace.csv'), ...
    sim_cpmg_matched_probe_img_rect(params));

disp(['Wrote imaging fixtures to ', fixture_dir]);

function write_imaging_fixture(path, kspace)
    [px, pz, ne] = size(kspace);
    table = zeros(numel(kspace), 5);
    row = 1;
    for ii = 1:px
        for jj = 1:pz
            for kk = 1:ne
                table(row, :) = [
                    ii, jj, kk, real(kspace(ii, jj, kk)), imag(kspace(ii, jj, kk))
                ];
                row = row + 1;
            end
        end
    end
    dlmwrite(path, table, 'precision', '%.17g');
end

function [echo_int_all] = sim_cpmg_matched_probe_img_rect(params)
    NE = params.NE;
    TE = params.TE;
    Tgrad = params.Tgrad;
    rho = params.rho;
    T1map = params.T1map;
    T2map = params.T2map;
    pxz = params.pxz;
    FOV = params.FOV;

    [sp, pp] = set_params_matched;
    T_90 = pp.T_90;
    T_180 = 2 * T_90;
    if pp.tacq > (TE - T_180)
        pp.tacq = TE - T_180;
    end

    sp.ny = 400;
    sp.maxoffs = 5;
    siz = size(rho);
    sp.nx = siz(1);
    sp.nz = siz(2);
    sp.rho = rho;
    sp.T1map = T1map;
    sp.T2map = T2map;
    px = pxz(1);
    pz = pxz(2);

    sp.plt_tx = 0;
    sp.plt_rx = 0;
    sp.plt_sequence = 0;
    sp.plt_axis = 0;
    sp.plt_mn = 0;
    sp.plt_echo = 0;
    sp.plt_output = 0;
    sp.plt_fields = 0;

    sp = create_fields_single_sided(sp);
    sp.w_1r = sp.w_1;

    tacq = (pi / 2) * pp.tacq / T_90;
    tdw = (pi / 2) * pp.tdw / T_90;
    nacq = round(tacq / tdw) + 1;
    tvect = linspace(-tacq / 2, tacq / 2, nacq).';
    isoc = exp(1i * tvect * sp.del_w);

    [C1, C2] = matching_network_design2(sp.L, sp.Q, sp.f0, sp.Rs, 0);
    sp.C1 = C1;
    sp.C2 = C2;

    Rtot = {};
    pp_in.tp = pp.T_90;
    pp_in.tdel = 2 * pp.T_90;
    pp_in.phi = pi / 2;
    pp_in.amp = 1;
    pp_out = calc_pulse_shape_matched_rect(sp, pp, pp_in);
    sp.tf1 = pp_out.tf1;
    sp.tf2 = pp_out.tf2;
    Rtot{1} = calc_rotation_matrix(sp, pp_out);

    pp_in.phi = 3 * pi / 2;
    pp_out = calc_pulse_shape_matched_rect(sp, pp, pp_in);
    Rtot{2} = calc_rotation_matrix(sp, pp_out);

    pp_in.tp = pp.T_180;
    pp_in.phi = 0;
    pp_out = calc_pulse_shape_matched_rect(sp, pp, pp_in);
    Rtot{3} = calc_rotation_matrix(sp, pp_out);

    pp_in.tp = pp.T_180;
    pp_in.phi = pi / 2;
    pp_out = calc_pulse_shape_matched_rect(sp, pp, pp_in);
    Rtot{4} = calc_rotation_matrix(sp, pp_out);

    texc = [pi / 2, -1];
    aexc = [1, 0];
    pexc1 = [1, 0];
    pexc2 = [2, 0];
    acq_exc = [0, 0];
    gexc = [0, 0];

    tenc = [(pi / 2) * Tgrad / T_90, pi, (pi / 2) * Tgrad / T_90];
    aenc = [0, 1, 0];
    penc1 = [0, 3, 0];
    penc2 = [0, 4, 0];
    acq_enc = [0, 0, 0];
    genc = [1, 0, 0];

    nref = 3;
    tref = zeros(1, nref * NE);
    pref1 = tref;
    pref2 = tref;
    aref = tref;
    acq_ref = tref;
    gref = tref;
    tfp = (pi / 2) * (TE - pp.T_180) / (2 * T_90);
    for ii = 1:NE
        tref((ii - 1) * nref + 1:ii * nref) = [tfp, pi, tfp];
        pref1((ii - 1) * nref + 1:ii * nref) = [0, 3, 0];
        pref2((ii - 1) * nref + 1:ii * nref) = [0, 4, 0];
        aref((ii - 1) * nref + 1:ii * nref) = [0, 1, 0];
        acq_ref((ii - 1) * nref + 1:ii * nref) = [0, 0, 1];
        gref((ii - 1) * nref + 1:ii * nref) = [0, 0, 0];
    end

    pp.tp = [texc, tenc, tref];
    pp.amp = [aexc, aenc, aref];
    pp.acq = [acq_exc, acq_enc, acq_ref];
    pp.grad = [gexc, genc, gref];
    pp.Rtot = Rtot;

    pul1 = [pexc1, penc1, pref1];
    pul2 = [pexc2, penc1, pref1];
    pul3 = [pexc1, penc2, pref2];
    pul4 = [pexc2, penc2, pref2];

    Tgradn = (pi / 2) * Tgrad / T_90;
    wxmax = pi * px^2 / (2 * FOV(1) * Tgradn);
    wzmax = pi * pz^2 / (2 * FOV(2) * Tgradn);
    gradx = wxmax * linspace(-1, 1, px);
    gradz = wzmax * linspace(-1, 1, pz);

    echo_int_all = zeros(px, pz, NE);
    for ii = 1:px
        spc = sp;
        ppc = pp;
        for jj = 1:pz
            spc.del_wg = gradx(ii) * spc.del_wx + gradz(jj) * spc.del_wz;

            ppc.pul = pul1;
            [~, mrx1] = calc_macq_matched_probe_relax4(spc, ppc);
            ppc.pul = pul2;
            [~, mrx2] = calc_macq_matched_probe_relax4(spc, ppc);
            mrx_x = mrx1 - mrx2;

            ppc.pul = pul3;
            [~, mrx3] = calc_macq_matched_probe_relax4(spc, ppc);
            ppc.pul = pul4;
            [~, mrx4] = calc_macq_matched_probe_relax4(spc, ppc);
            mrx_y = mrx3 - mrx4;

            echo_rx_x = isoc * mrx_x.';
            echo_rx_y = isoc * mrx_y.';
            echo_rx_xy = imag(echo_rx_x) - 1i * real(echo_rx_y);
            echo_int_all(ii, jj, :) = trapz(tvect, echo_rx_xy).';
        end
    end
end

function [pp_out] = calc_pulse_shape_matched_rect(sp, pp, pp_in)
    T_90 = pp.T_90;
    tdeln = (pi / 2) * pp_in.tdel / T_90;
    amp_zero = pp.amp_zero;

    pp_curr = pp;
    pp_curr.tp = [pp_in.tp, pp_in.tdel];
    pp_curr.phi = [pp_in.phi, 0];
    pp_curr.amp = [pp_in.amp, 0];

    sp.plt_rx = 0;
    [tvect, Icr, tf1, tf2] = find_coil_current(sp, pp_curr);
    pp_out.tf1 = tf1;
    pp_out.tf2 = tf2;

    delt = (pi / 2) * (tvect(2) - tvect(1)) / T_90;
    texc = delt * ones(1, length(tvect));
    pexc = atan2(imag(Icr), real(Icr));
    aexc = abs(Icr);
    aexc(aexc < amp_zero) = 0;

    pp_out.tp = [texc, -tdeln];
    pp_out.phi = [pexc, 0];
    pp_out.amp = [aexc, 0];
    pp_out.acq = zeros(1, length(texc) + 1);
end
