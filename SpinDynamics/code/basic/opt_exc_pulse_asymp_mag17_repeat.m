% Repeated optimization runs
% Design an excitation pulse with target number of segments

function opt_exc_pulse_asymp_mag17_repeat(ntarget,pulse_num,tacq)

% Use default SPA refocusing pulse
tref=pi*[3 0.14 0.72 0.14 3]; pref=pi*[0 1 0 1 0];
aref=[0 1 1 1 0];

% Uniform distribution of resonant offsets del_w0 (constant gradient)
maxoffs=20; numpts=2001;
del_w=linspace(-maxoffs,maxoffs,numpts);

% Calculate refocusing axis
[neff]=calc_rot_axis_arba3(tref,pref,aref,del_w,0);
params.neff=neff;
params.del_w=del_w;
params.tacq=tacq;

% Parameters of the excitation pulse
% Load original excitation pulse from load dat_files\results_mag_all.mat
tmp=load('dat_files\results_mag_all.mat'); results_sort = tmp.results_sort;
texc=results_sort{pulse_num,1}; pexc=results_sort{pulse_num,2};
nexc=length(texc);
params.ntarget=ntarget;

numstep=abs(ntarget-nexc);
results={};

% Repeat optimization till we have the right number of pulse segments
for count=1:numstep
    if count==1
        params.texc=texc;
        params.pexc=pexc;
    else
        params.texc=out.texc;
        params.pexc=out.pexc;
    end
    
    disp(length(params.pexc))
    
    % Run optimization
    [out]=opt_exc_pulse_asymp_mag17(params);
    
    results{count,1}=out.texc;
    results{count,2}=out.pexc;
    results{count,3}=tref;
    results{count,4}=pref;
    results{count,5}=out.echo_pk;
    results{count,6}=out.echo_rms;
    save(['results_mag17_' num2str(ntarget) '.mat'], 'results');
        
    disp(out.echo_pk)
    disp(out.echo_rms)
    disp(count)
end