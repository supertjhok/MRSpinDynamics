% TEST_2D
% Exercise the euclidean example with reduced two-dimensional test data.
%
% Purpose
%   Loads euclidean_data.mat, slices the example data to two dimensions, calls
%   euclidean, and prints closest/farthest points and distances.
%
% Inputs
%   Requires euclidean_data.mat in the current folder. That MAT-file is treated
%   as a local build fixture and is ignored by Git.
%
% Outputs
%   Prints closest/farthest coordinates, indices, and distances to the command
%   window.
%
% Key functions
%   euclidean.
%
% Notes
%   Used by the euclidean MEX/codegen examples.
% -------------------------------------------------------------------------

% Load the test data
load euclidean_data.mat

% Create 2-D versions of x and cb
x2d=x(1:2,:);
cb2d=cb(1:2,1:6:216);

% Determine closest and farthest points and corresponding distances
[y_min,y_max,idx,distance] = euclidean(x2d,cb2d);

% Display output for the closest point
disp('Coordinates of the closest point are: ');
disp(num2str(y_min'));
disp(['Index of the closest point is ', num2str(idx(1))]);
disp(['Distance to the closest point is ', num2str(distance(1))]);

disp(newline);

% Display output for the farthest point
disp('Coordinates of the farthest point are: ');
disp(num2str(y_max'));
disp(['Index of the farthest point is ', num2str(idx(2))]);
disp(['Distance to the farthest point is ', num2str(distance(2))]);
