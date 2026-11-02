function [x, fval, exitflag] = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options)
% FMINCON  Bound-constrained subset of fmincon (Octave fallback for the generators).
% Used only when the Optimization Toolbox / optim package is absent. Supports
% the bound-only calls in MATLABSpinDynamics (no linear or nonlinear
% constraints) through Octave's built-in sqp.
    if nargin < 7, lb = []; end
    if nargin < 8, ub = []; end
    if ~isempty(A) || ~isempty(b) || ~isempty(Aeq) || ~isempty(beq) ...
            || (nargin >= 9 && ~isempty(nonlcon))
        error('fmincon fallback supports bound constraints only');
    end
    maxiter = 1000;
    tol = 1e-12;
    if nargin >= 10 && ~isempty(options)
        if isfield(options, 'MaxIter') && ~isempty(options.MaxIter)
            maxiter = options.MaxIter;
        end
    end
    shape = size(x0);
    % Finite stand-ins: Octave sqp drops infinite bounds and then compares
    % the remaining lower and upper bounds, which fails for mixed cases.
    if isempty(lb), lb = -realmax * ones(numel(x0), 1); end
    if isempty(ub), ub = realmax * ones(numel(x0), 1); end
    objective = @(v) fun(reshape(v, shape));
    [v, fval, info] = sqp(x0(:), objective, [], [], lb(:), ub(:), maxiter, tol);
    x = reshape(v, shape);
    exitflag = double(info == 101 || info == 104);
end
