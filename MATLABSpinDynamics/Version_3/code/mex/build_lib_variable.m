% BUILD_LIB_VARIABLE
% Generate a variable-size C library for the euclidean example.
%
% Purpose
%   Loads euclidean_data.mat, declares bounded variable-size inputs with
%   coder.typeof, and runs codegen to build a C library from euclidean.m.
%
% Inputs
%   Requires euclidean_data.mat in the current folder. That MAT-file is treated
%   as a local build fixture and is ignored by Git.
%
% Outputs
%   Creates MATLAB Coder output under codegen and related local build products.
%
% Key functions
%   coder.typeof, codegen, euclidean.
%
% Notes
%   Requires MATLAB Coder and a configured C/C++ compiler. Generated outputs are
%   ignored release artifacts.
% -------------------------------------------------------------------------

% Load the test data
load euclidean_data.mat

% Use coder.typeof to specify variable-size inputs
eg_x=coder.typeof(x,[3 1],1);
eg_cb=coder.typeof(cb,[3 216],1);

% Generate code for euclidean.m using coder.typeof to specify
% upper bounds for the example inputs
codegen -report -config:lib euclidean.m -args {eg_x,eg_cb}
