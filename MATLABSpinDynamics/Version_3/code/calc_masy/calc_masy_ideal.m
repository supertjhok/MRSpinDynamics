% CALC_MASY_IDEAL
% Calculate CPMG asymptotic magnetization with no probe-circuit effects.
%
% Signature
%   masy = calc_masy_ideal(sp,pp)
%
% Inputs
%   sp - Simulation/system parameter structure. Required fields include
%     del_w and plotting flags such as plt_axis and plt_rx.
%   pp - Pulse-sequence parameter structure. Required fields include T_90,
%     tref, pref, aref, texc, pexc, aexc, tacq, and tcorr.
%
% Outputs
%   masy - Complex asymptotic magnetization spectrum after phase cycling.
%
% Dependencies
%   calc_rot_axis_arba3, sim_spin_dynamics_asymp_mag3.
%
% Notes
%   This ideal workflow converts absolute pulse timings in pp to normalized
%   time using T_90 before calling the lower-level spin-dynamics routines.
%
% Written by: Soumyajit Mandal, 03/28/19
% --------------------------------------------------------------

function [masy]=calc_masy_ideal(sp,pp)

T_90=pp.T_90; % Rectangular T_90 time

% Convert acquisition time to normalized time
tacq=(pi/2)*pp.tacq/T_90;

% Calculate refocusing axis
tref=(pi/2)*pp.tref/T_90; % Convert to normalized time
pref=pp.pref;
aref=pp.aref;
[neff]=calc_rot_axis_arba3(tref,pref,aref,sp.del_w,sp.plt_axis);

% Include timing correction for excitation pulse
texc=(pi/2)*pp.texc/T_90; % Convert to normalized time
texc=[texc -1]; % Include timing correction
pexc=[pp.pexc 0]; 
aexc=[pp.aexc 0];

[masy1]=sim_spin_dynamics_asymp_mag3(texc,pexc,aexc,neff,sp.del_w,tacq);
[masy2]=sim_spin_dynamics_asymp_mag3(texc,pexc+pi,aexc,neff,sp.del_w,tacq);
masy=(masy1-masy2)/2;

if sp.plt_rx
    figure;
    plot(sp.del_w,real(masy),'b-'); hold on;
    plot(sp.del_w,imag(masy),'r-');
    title('Asymptotic magnetization')
    xlabel('\Delta\omega_{0}/\omega_{1,max}')
    ylabel('M_{asy}')
    set(gca,'FontSize',14);   
end

