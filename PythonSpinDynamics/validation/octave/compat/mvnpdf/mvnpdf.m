function p = mvnpdf(x, mu, sigma)
% MVNPDF  Multivariate normal density (Octave fallback for the generators).
% Used only when the Statistics Toolbox / statistics package is absent;
% matches mvnpdf(X, MU, SIGMA) for row observations X and a full covariance.
    d = size(x, 2);
    centred = x - mu;
    p = exp(-0.5 * sum((centred / sigma) .* centred, 2)) ...
        / sqrt((2 * pi)^d * det(sigma));
end
