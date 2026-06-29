% BUILD_LIB_FIXED
% Generate a fixed-size C library for the euclidean example.
%
% Purpose
%   Loads euclidean_data.mat and runs codegen with the example inputs to build
%   a C library from euclidean.m.
%
% Inputs
%   Requires euclidean_data.mat in the current folder. That MAT-file is treated
%   as a local build fixture and is ignored by Git.
%
% Outputs
%   Creates MATLAB Coder output under codegen and related local build products.
%
% Key functions
%   codegen, euclidean.
%
% Notes
%   Requires MATLAB Coder and a configured C/C++ compiler. Generated outputs are
%   ignored release artifacts.
% -------------------------------------------------------------------------

% Load the test data
load euclidean_data.mat
% Generate code for euclidean.m with codegen. Use the test data as example
% input
codegen -report -config:lib euclidean.m -args {x, cb}
