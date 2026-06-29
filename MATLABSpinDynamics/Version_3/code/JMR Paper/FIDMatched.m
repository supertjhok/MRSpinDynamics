% FIDMATCHED
% Run the matched-probe FID workflow used by the JMR paper scripts.
%
% Purpose
%   Builds matched-probe FID parameters and calls the matched FID simulation
%   helper.
%
% Inputs
%   This script takes no function arguments. Parameters are created by
%   set_params_matched_FID.
%
% Outputs
%   Leaves sp, pp, and echo_int_all in the workspace.
%
% Key functions
%   set_params_matched_FID, FID_Matched_Fun.
%
% Notes
%   This is a historical paper workflow. Confirm parameter choices before using
%   it as a public example.
% -------------------------------------------------------------------------

[sp, pp]=set_params_matched_FID;

% ----------------------------------------------
% Run simulation
% ----------------------------------------------
[echo_int_all]=FID_Matched_Fun(sp,pp);
