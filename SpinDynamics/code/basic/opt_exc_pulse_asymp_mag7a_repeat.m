% Repeated optimization runs, store current maximum
% Gradually make excitation pulse longer (by adding segments)
% Re-optimize earlier results

function opt_exc_pulse_asymp_mag7a_repeat(tref,pref,aref,delt)

% Load original results
tmp=load('dat_files\results_mag7.mat');
results_orig=tmp.results;

% Particularly promising pulses from results_mag7.mat
expnums=[16,22,28,30,34,48,51];

[neff,del_w]=calc_rot_axis_arba(tref,pref,aref);

results={};
for i=1:length(expnums)
    texc=results_orig{expnums(i),1}; pexc=results_orig{expnums(i),2};
    [texc,pexc,echo_pk,echo_rms]=opt_exc_pulse_asymp_mag7a(neff,del_w,delt,texc,pexc);
    nseg=length(texc);
    
    results{i,1}=texc;
    results{i,2}=pexc;
    results{i,3}=echo_pk;
    results{i,4}=echo_rms;
    save results_mag8.mat results
    
    disp(nseg)
    disp(echo_pk)
    disp(echo_rms)
end