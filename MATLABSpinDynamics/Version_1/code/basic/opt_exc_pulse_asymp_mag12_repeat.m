% Repeated optimization runs, store current maximum
% Gradually make excitation pulse shorter (by reducing number of segments)
% Use Colm's optimization code for greater speed

function opt_exc_pulse_asymp_mag12_repeat(tref,pref,aref,delt)

delta=1; % Segments to remove at each step
pulse_num=46;

% Load initial excitation pulse
tmp=load('dat_files\results_mag11_reopt.mat');
results_sort=tmp.results;
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
    
    [out]=opt_exc_pulse_asymp_mag12(params);
        
    results{count,1}=out.texc;
    results{count,2}=out.pexc;
    results{count,3}=out.echo_pk;
    results{count,4}=out.echo_rms;
    save results_mag12.mat results
    
    % Shorten excitation pulse and repeat optimization
    count=count+1;
    
    disp(out.echo_pk)
    disp(out.echo_rms)
    disp(length(out.texc))
end