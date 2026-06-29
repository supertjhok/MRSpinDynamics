% Repeated optimization runs, store current maximum
% Gradually make excitation pulse shorter (by making segments shorter)
% Re-introduce amplitude modulation into previously optimized pulses

function opt_exc_pulse_asymp_mag6_repeat(tref,pref,aref,delt)

% Load optimization results
tmp=load('dat_files\results_mag5.mat');
results_orig=tmp.results;
sizres=size(results_orig);

[neff,del_w]=calc_rot_axis_arba(tref,pref,aref);

results={};
for i=1:sizres(1)
    texc=results_orig{i,1}; pexc=results_orig{i,2};
    
    disp(results_orig{i,3})
    disp(results_orig{i,4})
    
    [aexc,echo_pk,echo_rms]=opt_exc_pulse_asymp_mag6(neff,del_w,delt,texc,pexc);
    
    results{i,1}=texc;
    results{i,2}=pexc;
    results{i,3}=aexc;
    results{i,4}=echo_pk;
    results{i,5}=echo_rms;
    save results_mag6.mat results
    
    disp(echo_pk)
    disp(echo_rms)
end