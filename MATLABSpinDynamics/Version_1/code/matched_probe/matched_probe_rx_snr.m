% Calculate matched filter output and SNR for an optimum receiver
% ------------------------------------------------------
% Written by: Soumyajit Mandal, 09/09/19

function [SNR]=matched_probe_rx_snr(sp,mrx,pnoise,f)

mf_type=sp.mf_type;
del_w=sp.del_w;

% Matched filtering (maximize SNR at t=0)
switch mf_type
    case 1 % Matched filter (for white noise)
        mf=conj(mrx);
    case 2 % Matched filter (for colored noise)
        mf=conj(mrx)./pnoise;
end
mf=mf/sqrt(trapz(del_w,abs(mf).^2)); % Normalize matched filter amplitude

vsig=trapz(del_w,mrx.*mf); % Signal amplitude (at t=0)
vnoise=sqrt(trapz(f,pnoise.*abs(mf).^2)); % Total integrated noise (V_rms)
SNR=real(vsig)/vnoise; % Signal-to-noise ratio (SNR) per scan in voltage units
SNR=SNR/1e8;