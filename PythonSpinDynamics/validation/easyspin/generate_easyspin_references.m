function generate_easyspin_references(easyspin_label)
% GENERATE_EASYSPIN_REFERENCES
% Write EasySpin reference data for tests/test_easyspin_references.py.
%
% Needs MATLAB and EasySpin 6 on the path (addpath('<EasySpin>/easyspin')).
% Run from PythonSpinDynamics:
%   matlab -batch "addpath('<EasySpin>/easyspin'); addpath('validation/easyspin'); generate_easyspin_references"
% Outputs go to validation/external/easyspin/ and are committed, so the
% comparison test runs without MATLAB. easyspin_label records the EasySpin
% release when the installed copy does not report one (source checkouts).

if exist('pepper', 'file') ~= 2 || exist('resfields', 'file') ~= 2
    error('EasySpin (pepper, resfields) is not on the MATLAB path');
end
root = fileparts(fileparts(mfilename('fullpath')));
out = fullfile(root, 'external', 'easyspin');
if exist(out, 'dir') ~= 7
    mkdir(out);
end

meta = struct();
meta.easyspin_version = easyspin_version_string();
if nargin > 0 && ~isempty(easyspin_label)
    meta.easyspin_version = easyspin_label;
end
meta.matlab_version = version;
meta.generated = datestr(now, 'yyyy-mm-dd');

% 1. Powder spectra of a rhombic g tensor, with and without the field-swept
%    1/g (Aasa-Vanngard) intensity factor.
Sys = struct('S', 1/2, 'g', [1.95 2.05 2.20], 'lw', [1.0 0]);  % lw: Gaussian FWHM, mT
Exp = struct('mwFreq', 9.5, 'Range', [295 360], 'nPoints', 2601, 'Harmonic', 0);
for freq2field = [1 0]
    % 0.5-degree knots with 4-fold spline refinement: converged for 1 mT lines.
    Opt = struct('GridSize', [181 4], 'Freq2Field', freq2field);
    [field_mt, spectrum] = pepper(Sys, Exp, Opt);
    name = sprintf('powder_rhombic_freq2field%d.csv', freq2field);
    dlmwrite(fullfile(out, name), [field_mt(:), spectrum(:)], 'precision', '%.17g');
end
meta.powder = struct('g', Sys.g, 'lw_gaussian_fwhm_mT', Sys.lw(1), ...
    'mwFreq_GHz', Exp.mwFreq, 'range_mT', Exp.Range, 'nPoints', Exp.nPoints, ...
    'GridSize', [181 4]);

% 2. Exact (matrix-diagonalization) resonance fields with isotropic g and
%    hyperfine couplings, including the Breit-Rabi regime of the hydrogen atom.
cases = {
    'h_atom',     2.00226, '1H',      1420.40575;
    'proton_50',  2.0036,  '1H',      50.0;
    'n14_45',     2.0058,  '14N',     45.0;
    'h_n14',      2.0050,  '1H,14N',  [30.0 45.0];
};
Exp = struct('mwFreq', 9.5, 'Range', [100 600], 'SampleFrame', [0 0 0]);
meta.hyperfine = struct('mwFreq_GHz', Exp.mwFreq, 'range_mT', Exp.Range, 'cases', {{}});
for k = 1:size(cases, 1)
    Sys = struct('S', 1/2, 'g', cases{k, 2}, 'Nucs', cases{k, 3}, 'A', cases{k, 4});
    % Threshold 0 keeps every transition, including forbidden lines near the
    % default 1e-4 selection threshold. RediagLimit 1 rediagonalizes at every
    % resonance field; the default 0.95 interpolates eigenvectors between
    % knots, which leaves the weak forbidden-line amplitudes ~10% high.
    [positions_mt, amplitudes] = resfields(Sys, Exp, struct('Threshold', 0, 'RediagLimit', 1));
    rows = [positions_mt(:), amplitudes(:)];
    rows = sortrows(rows, 1);
    dlmwrite(fullfile(out, sprintf('resfields_%s.csv', cases{k, 1})), rows, 'precision', '%.17g');
    meta.hyperfine.cases{end + 1} = struct('name', cases{k, 1}, 'g', cases{k, 2}, ...
        'nucs', cases{k, 3}, 'A_MHz', cases{k, 4});
end

fid = fopen(fullfile(out, 'metadata.json'), 'w');
fprintf(fid, '%s\n', jsonencode(meta, 'PrettyPrint', true));
fclose(fid);
disp(['Wrote EasySpin references to ', out]);
end

function value = easyspin_version_string()
    try
        info = easyspin('info');
        value = info.Version;
    catch
        value = 'unknown';
    end
end
