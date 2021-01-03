% Repeated optimization runs, store current maximum
% Gradually make excitation pulse longer (by adding segments)
% Use Colm's optimization code for greater speed
% Re-optimize earlier results

function opt_exc_pulse_asymp_mag8a_repeat(tref,pref,aref,delt)

% Load original results
tmp=load('dat_files\results_mag9.mat');
results_orig=tmp.results;

% Particularly promising pulses from results_mag9.mat
expnums=[5,20,24,29,31,41];

results={};
for i=1:length(expnums)
    texc=results_orig{expnums(i),1}; pexc=results_orig{expnums(i),2};
    [texc,pexc,echo_pk,echo_rms]=opt_exc_pulse_asymp_mag8a(texc,pexc,tref,pref,aref,delt);
    nseg=length(texc);
    
    results{i,1}=texc;
    results{i,2}=pexc;
    results{i,3}=echo_pk;
    results{i,4}=echo_rms;
    save results_mag10.mat results
    
    disp(nseg)
    disp(echo_pk)
    disp(echo_rms)
end