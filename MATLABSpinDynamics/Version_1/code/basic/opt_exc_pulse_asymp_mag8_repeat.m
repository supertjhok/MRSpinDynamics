% Repeated optimization runs, store current maximum
% Gradually make excitation pulse longer (by adding segments)
% Use Colm's optimization code for greater speed

function opt_exc_pulse_asymp_mag8_repeat(tref,pref,aref,delt)

delta=2; % Number of segments to add at each step

% Load initial excitation pulse (100 segments)
tmp=load('dat_files\results_mag5.mat');
results=tmp.results;
texc=results{11,1}; pexc=results{11,2};

count=1;

results={};
while(1)
    [texc,pexc,echo_pk,echo_rms]=opt_exc_pulse_asymp_mag8(texc,pexc,tref,pref,aref,delta,delt);
    nseg=length(texc);
    
    results{count,1}=texc;
    results{count,2}=pexc;
    results{count,3}=echo_pk;
    results{count,4}=echo_rms;
    save results_mag9.mat results
    
    disp(nseg)
    disp(echo_pk)
    disp(echo_rms)
    
    % Lengthen excitation pulse and repeat optimization
    count=count+1;
end