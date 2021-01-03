% Calculate time domain echo from arbitrary magnetization
% Use zero-filling to get smoother echo shapes
% ------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19
% Modified for arbitary magnetization, 09/09/19
% Precalculate ischromat vector for speed, 09/17/19

% wvect -> offset frequency
% tacq -> acquisition window length, tdw -> dwell time
function [echo]=calc_time_domain_echo_arb2(mrx,sp)

% Calculate echo
echo=(sp.isoc*mrx')'; % Size: [1,nacq]

if sp.plt_echo
    T_180=pi; % Normalized T_180 time
    figure;
    plot(tvect/T_180,real(echo),'b-'); hold on;
    plot(tvect/T_180,imag(echo),'r-');
    % plot(tvect/T_180,abs(echo),'k-');
    set(gca,'FontSize',14);
    xlabel('Normalized time, t / T_{180}');
    title('Time-domain echo');
    xlim([-3 3]);
end