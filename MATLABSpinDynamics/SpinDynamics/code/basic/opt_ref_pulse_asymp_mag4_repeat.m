% Repeated optimization runs of previously optimized pulse

function opt_ref_pulse_asymp_mag4_repeat(filname,pulse_nums,tacq)

nums=length(pulse_nums);

% Load refocusing pulse to re-optimize
tmp=load(filname); results_orig=tmp.results;

% Uniform distribution of resonant offsets del_w0 (constant gradient)
maxoffs=20; numpts=2001;
params.del_w=linspace(-maxoffs,maxoffs,numpts);
params.opt_window=hamming(numpts)';
params.tacq=tacq;

% Parameters of the refocusing pulse, assuming internal anti-symmetry
params.tmin=0.05*pi; params.tmax=0.25*pi;
params.tfp=3*pi; % Free precession time

results={};
for i=1:nums
    % Get new initial conditions
    params.tref=results_orig{pulse_nums(i),1}; params.pref=results_orig{pulse_nums(i),2};
    params.aref=results_orig{pulse_nums(i),3}; nref=(length(params.tref)-3)/2+1;
    params.tref=params.tref(2:nref+1); params.pref=params.pref(2:nref+1);
    params.aref=params.aref(2:nref+1);
    
    % Run optimization
    [out]=opt_ref_pulse_asymp_mag4(params);
    tref=out.tref; pref=out.pref; aref=out.aref;
    
    results{i,1}=[params.tfp tref fliplr(tref(1:nref-1)) params.tfp]; % Store refocusing cycle
    results{i,2}=[0 pref -fliplr(pref(1:nref-1)) 0];
    results{i,3}=[0 aref aref(1:nref-1) 0];
    results{i,4}=out.axis_rms;
    save(['results_ref_mag4_' num2str(nref) '.mat'], 'results');
    
    disp(out.axis_rms)
    disp(i)
end