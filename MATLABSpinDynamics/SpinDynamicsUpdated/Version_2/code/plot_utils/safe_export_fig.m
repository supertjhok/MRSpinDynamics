function did_export = safe_export_fig(filename)
% SAFE_EXPORT_FIG Export the current figure when export_fig is available.
%
% Signature
%   did_export = safe_export_fig(filename)
%
% Inputs
%   filename - Output file path passed to export_fig.
%
% Outputs
%   did_export - True when export_fig was found and called; false otherwise.
%
% Notes
%   Several historical scripts used unconditional export_fig calls to absolute
%   personal paths. This helper keeps those scripts runnable on machines
%   without export_fig while preserving export behavior when the dependency is
%   installed.
% -------------------------------------------------------------------------

did_export = false;

if exist('export_fig', 'file') ~= 2
    warning('MATLABSpinDynamics:MissingExportFig', ...
        'Skipping figure export because export_fig is not on the MATLAB path: %s', filename);
    return;
end

output_dir = fileparts(filename);
if ~isempty(output_dir) && exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

export_fig(filename);
did_export = true;
end
