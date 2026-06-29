% Repeated optimization runs, store current maximum
% Gradually make excitation pulse longer (by adding segments)

function opt_exc_pulse_asymp_mag7_repeat(tref,pref,aref,delt)

delta=2; % Number of segments to add at each step

% Load initial excitation pulse (100 segments)
tmp=load('dat_files\results_mag5.mat');
results=tmp.results;
texc=results{11,1}; pexc=results{11,2};

count=1;
[neff,del_w]=calc_rot_axis_arba(tref,pref,aref);

results={};
while(1)
    [texc,pexc,echo_pk,echo_rms]=opt_exc_pulse_asymp_mag7(neff,del_w,delt,texc,pexc,delta);
    nseg=length(texc);
    
    results{count,1}=texc;
    results{count,2}=pexc;
    results{count,3}=echo_pk;
    results{count,4}=echo_rms;
    save results_mag7.mat results
    
    disp(nseg)
    disp(echo_pk)
    disp(echo_rms)
    
    % Lengthen excitation pulse and repeat optimization
    count=count+1;
end