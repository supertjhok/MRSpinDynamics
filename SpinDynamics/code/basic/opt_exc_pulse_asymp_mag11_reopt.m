% Repeated optimization runs, store current maximum
% Re-optimize phases of existing pulses

function opt_exc_pulse_asymp_mag11_reopt(tref,pref,aref,delt)

% Load original excitation pulses
tmp=load('results_mag12.mat');
results_old=tmp.results;
tmp=size(results_old); nump=tmp(1);

[neff,del_w]=calc_rot_axis_arba(tref,pref,aref);
params.neff=neff;
params.del_w=del_w;
params.delt=delt;

% Load current set of re-optimized excitation pulses
tmp=load('results_mag12_reopt.mat');
results=tmp.results;
tmp=size(results); nums=tmp(1);

for i=nums+1:nump
    params.texc=results_old{i,1};
    params.pexc=results_old{i,2};
      
    [out]=opt_exc_pulse_asymp(params);
        
    results{i,1}=out.texc;
    results{i,2}=out.pexc;
    results{i,3}=out.echo_pk;
    results{i,4}=out.echo_rms;
    save results_mag12_reopt.mat results
    
    disp(out.echo_pk)
    disp(out.echo_rms)
    disp(i)
end