% Repeated optimization runs, store current maximum
% Gradually make excitation pulse shorter

function opt_exc_pulse_asymp_mag4_repeat(tref,pref,aref,delt)

delta=2; % Number of segments to shorten at each step

% Load initial excitation pulse (100 segments)
tmp=load('results_mag3_rms_2.mat');
start=tmp.pexc;

% Load initial excitation pulse (current)
%tmp=load('results_mag4.mat');
%results=tmp.results;
%sizres=size(results);
%start=results{sizres(1),2};

nseg=length(start)-delta;

count=1;
[neff,del_w]=calc_rot_axis_arba(tref,pref,aref);

results={};
while(1)
    [texc,pexc,echo_pk,echo_rms]=opt_exc_pulse_asymp_mag4(neff,del_w,delt,nseg,start);
    nseg=length(texc);
    
    results{count,1}=texc;
    results{count,2}=pexc;
    results{count,3}=echo_pk;
    results{count,4}=echo_rms;
    save results_mag4.mat results
    
    disp(nseg)
    disp(echo_pk)
    disp(echo_rms)
    
    % Shorten excitation pulse and repeat optimization
    nseg=nseg-delta;
    start=pexc;
    count=count+1;
end
