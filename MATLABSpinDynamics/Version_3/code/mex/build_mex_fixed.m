% BUILD_MEX_FIXED
% Generate and test a fixed-size MEX function for the euclidean example.
%
% Purpose
%   Loads euclidean_data.mat and runs codegen with example inputs to build and
%   validate euclidean_mex.
%
% Inputs
%   Requires euclidean_data.mat and test.m in the current folder. The MAT-file
%   is treated as a local build fixture and is ignored by Git.
%
% Outputs
%   Creates euclidean MEX/codegen outputs locally.
%
% Key functions
%   codegen, euclidean, test.
%
% Notes
%   Requires MATLAB Coder and a configured C/C++ compiler. Generated outputs are
%   ignored release artifacts.
% -------------------------------------------------------------------------

% Load the test data
load euclidean_data.mat
% Generate code for euclidean.m with codegen. Use the test data as example
% input. Validate MEX by using test.m.
codegen -report euclidean.m -args {x, cb} -test test
