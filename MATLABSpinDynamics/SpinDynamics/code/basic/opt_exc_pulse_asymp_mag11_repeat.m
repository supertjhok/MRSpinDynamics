% Repeated optimization runs, store current maximum
% Gradually make excitation pulse longer (by stretching segments)
% Use Colm's optimization code for greater speed

function opt_exc_pulse_asymp_mag11_repeat(tref,pref,aref,delt)

delta=0.02; % Fraction to lengthen at each step
pulse_num=47;

% Load initial excitation pulse (100 segments)
tmp=load('dat_files\results_mag_all.mat');
results_sort=tmp.results_sort;
out.texc=results_sort{pulse_num,1}; out.pexc=results_sort{pulse_num,2};

count=1;
[neff,del_w]=calc_rot_axis_arba(tref,pref,aref);
params.neff=neff;
params.del_w=del_w;
params.delt=delt;
params.delta=delta;

results={};
while(1)
    params.texc=out.texc;
    params.pexc=out.pexc;
    
    [out]=opt_exc_pulse_asymp_mag11(params);
        
    results{count,1}=out.texc;
    results{count,2}=out.pexc;
    results{count,3}=out.echo_pk;
    results{count,4}=out.echo_rms;
    save results_mag11.mat results
    
    % Lengthen excitation pulse and repeat optimization
    count=count+1;
    
    disp(out.echo_pk)
    disp(out.echo_rms)
    disp(count)
end