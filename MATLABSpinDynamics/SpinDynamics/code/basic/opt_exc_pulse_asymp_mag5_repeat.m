% Repeated optimization runs, store current maximum
% Gradually make excitation pulse shorter (by making segments shorter)

function opt_exc_pulse_asymp_mag5_repeat(tref,pref,aref,delt)

delta=0.02; % Fraction of shortening at each step
tscale=1-delta;

% Load initial excitation pulse (100 segments)
tmp=load('results_mag3_rms_2.mat');
start=tmp.pexc;

count=1;
[neff,del_w]=calc_rot_axis_arba(tref,pref,aref);

results={};
while(1)
    [texc,pexc,echo_pk,echo_rms]=opt_exc_pulse_asymp_mag5(neff,del_w,delt,tscale,start);
    nseg=length(texc);
    
    results{count,1}=texc;
    results{count,2}=pexc;
    results{count,3}=echo_pk;
    results{count,4}=echo_rms;
    save results_mag5.mat results
    
    disp(tscale)
    disp(echo_pk)
    disp(echo_rms)
    
    % Shorten excitation pulse and repeat optimization
    %tscale=tscale-delta; % Linear decrease
    tscale=tscale*(1-delta); % Geometric decrease
    start=pexc;
    count=count+1;
end