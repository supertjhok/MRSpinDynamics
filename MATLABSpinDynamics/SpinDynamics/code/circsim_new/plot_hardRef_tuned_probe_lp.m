% Calculate rotation axis of a refocusing cycle including transmitter and
% receiver bandwidth effects
% --------------------------------------------------------------
% params = [tref, pref, aref, tfp, tacq, Rs(Qsw_on,Qsw_off,Tx_on)] (all times normalized to w1 = 1)
% --------------------------------------------------------------

% function [n,SNR] = plot_opt_ref_results_tuned_probe_lp(results_num,pulse_num)
% vars: sens - coil sensitivity, rat - 90/180 ratio, len - normalized pulse
% length

function [masy,SNR] = plot_hardRef_tuned_probe_lp(vars)

[sp,pp]=set_params_lp_tapped;
sp.sens = vars.sens;
% tmp= load('results_circsim_Sep10_6.mat');
% tmp = load('results_circsim_Sept12_nexc_160.mat');
% results = tmp.results;
% sp = results{1,9};
% pp = results{1,10};
sp.cycle=2*pi/sp.w0; % Length of one RF cycle

sp.plt_axis=0; % plot rotation axis as a function of delta w
sp.plt_tx=1; % plot current in coil
sp.plt_rx=0; % plot circuit properties

params.tacq=4*pi; 
params.Rs=[sp.Ron 10^6 sp.Roff];
params.tqs = 1*sp.cycle*(pi/2)/pp.T_90; % Delay between pulse end & Q-switch turn on = 1 RF cycle
params.pcycle=0;
B1max=(pi/2)/(pp.T_90*sp.gamma);
pp.N = 16;

%sp.plt_axis=0;  sp.plt_tx=1; sp.plt_rx=0; 

% Define refocusing pulse
% params.tfp=3*pi;
techo = vars.techo;
params.tfp=(techo - pi)/2;
nref = 10; % Each segment = T_180/10
lref = vars.len*vars.rat*(pi/2)/nref;
params.tref=lref*ones(1,nref);
params.pref=pi*ones(1,nref);
params.aref=ones(1,nref);
% params.tref = pi;
% params.pref = pi;
% params.aref = 1;

% Create refocusing cycle
pp.tref=[params.tfp params.tref params.tqs (params.tfp-params.tqs)]*pp.T_90/(pi/2);
pp.pref=[0 params.pref 0 0]; pp.aref=[0 params.aref 0 0]; 
pp.Rsref=[params.Rs(1) params.Rs(3)*ones(1,length(params.tref)) params.Rs(2) params.Rs(1)];

[~,~,tvect,Icr] = tuned_probe_lp(sp,pp);

delt=(pi/2)*(tvect(2)-tvect(1))/pp.T_90; % Convert to normalized time
trefc=delt*ones(1,length(tvect));
prefc=atan2(imag(Icr),real(Icr));
arefc=abs(Icr)*sp.sens/B1max;
arefc(arefc<pp.amp_zero)=0; % Threshold amplitude

if sp.plt_tx
    figure(11); hold on;
    plot(tvect*1e6,arefc,'r');
    xlabel('Time (\mus)')
    ylabel('Normalized current amplitude in coil (refocusing)')
end
 
[n,SNRdummy]=calc_rot_axis_tuned_probe_lp(params,sp,pp);
params.neff =n ;

% Define excitation pulse
% Note: add extra delay at end to allow excitation pulse to ring down
% nexc = 5;
% lexc = vars.len*(pi/2)/nexc;
% params.texc=lexc*ones(1,nexc+1);
% params.pexc=pi/2*ones(1,nexc+1);
% params.aexc=[ones(1,nexc) 0];
params.texc = [vars.len*pi/2 pi/2];
params.pexc = [pi/2 pi/2];
params.aexc = [1 0];

% Create excitation cycle
pp.tref=params.texc*pp.T_90/(pi/2);
pp.pref=params.pexc; pp.aref=params.aexc; 
% pp.Rsref=params.Rs(3)*ones(1,length(params.pexc));
% if length(params.Rs) == 3
%     pp.Rsref=params.Rs(3)*ones(1,length(params.pexc));
% elseif length(params.Rs) == 2
%     pp.Rsref=params.Rs(2)*ones(1,length(params.pexc));
% end
pp.Rsref = [params.Rs(3) params.Rs(1)];
amp_zero=pp.amp_zero; % Minimum amplitude for calculations

sp.plt_tx=2; % plot current in coil
[~,~,tvect,Icr] = tuned_probe_lp(sp,pp);

delt=(pi/2)*(tvect(2)-tvect(1))/pp.T_90; % Convert to normalized time
texc=delt*ones(1,length(tvect));
pexc=atan2(imag(Icr),real(Icr));
aexc=abs(Icr)*sp.sens/B1max;
aexc(aexc<amp_zero)=0; % Threshold amplitude

if sp.plt_tx
    figure(12); hold on;
    plot(tvect*1e6,aexc,'r');
    xlabel('Time (\mus)')
    ylabel('Normalized current amplitude in coil (excitation)')
end

% Add timing correction for the excitation pulse
% ----------------------------------------------------------------------
% timing correction = -(params.texc(2)+2*params.texc(1)/pi)
% first term: removes the added ringdown delay at the end of the pulse
% second term: removes effects of precession during the pulse (to first order)
% ----------------------------------------------------------------------
texc=[texc -(params.texc(2)+2*params.texc(1)/pi)];
pexc=[pexc 0]; aexc=[aexc 0];

% Calculate spin dynamics
[masy]=sim_spin_dynamics_asymp_mag3(texc,pexc,aexc,params.neff,sp.del_w,params.tacq);
[~,SNR]=tuned_probe_rx(sp,pp,masy); % Filtering by tuned receiver
SNR=SNR/1e8;

% [mrx,SNR,SNR_max, masy] = calc_masy_tuned_probe_lp(params,sp,pp);
