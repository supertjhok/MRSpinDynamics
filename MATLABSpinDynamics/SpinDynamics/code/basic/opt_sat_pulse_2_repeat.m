function opt_sat_pulse_2_repeat

% Simulation parameters
%numseg=50; lseg=0.1*(pi/2);
numpts=8192;  maxoffs=20;

%params.tp=lseg*ones(1,numseg);
params.del_w=linspace(-maxoffs,maxoffs,numpts);
params.wtfun=hamming(numpts)';

tmp=load('results_sat_pulse_1.mat'); results_orig=tmp.results;

count=1;
results={};
while(1)
    params.phi=results_orig{count}.phi;
    params.tp=results_orig{count}.tp;
    [out]=opt_sat_pulse_2(params);
    
    results{count}=out;
    save results_sat_pulse_2.mat results
    
    disp(out.sat)
    disp(count)
    count=count+1;
end