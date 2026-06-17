# MATLAB Help Comment Standard

Use this format for active scripts in `SpinDynamicsUpdated/Version_2/code`.
The help block should be the first nonblank content in the file so MATLAB's
`help script_name` command can show useful guidance.

```matlab
% SCRIPT_NAME
% Brief one-line purpose.
%
% Purpose
%   Explain what workflow the script runs and what result it produces.
%
% Inputs
%   This script takes no function arguments. List required files, images,
%   MAT-files, or user prompts here.
%
% Outputs
%   List figures, variables left in the workspace, result files, or prompts.
%
% Key functions
%   List the main parameter constructors, simulators, or plotting routines.
%
% Notes
%   Mention assumptions, runtime expectations, path requirements, or known
%   limitations.
% -------------------------------------------------------------------------
```

Use this format for active function files. Keep the signature synchronized with
the function declaration and list structure fields when a function expects an
`sp`, `pp`, `params`, or transfer-function structure.

```matlab
% FUNCTION_NAME
% Brief one-line purpose.
%
% Signature
%   [out1,out2] = function_name(in1,in2)
%
% Inputs
%   in1 - Type/shape and meaning.
%   in2 - Type/shape and meaning.
%
% Outputs
%   out1 - Type/shape and meaning.
%   out2 - Type/shape and meaning.
%
% Dependencies
%   List major helper functions or required parameter constructors.
%
% Notes
%   Mention normalization, units, plotting side effects, performance issues,
%   or legacy/version relationships.
% -------------------------------------------------------------------------
```

For MATLAB functions, the help block should be the first comment block in the
file. If the function already has useful provenance or algorithm-history notes,
keep them after the standardized argument summary.
