% GENERATE_OPTIMIZATION_RESULT_FIXTURES
% Generate compact MATLAB optimizer-output fixtures for workflow validation.

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
generate_optimization_result_fixtures_main();
