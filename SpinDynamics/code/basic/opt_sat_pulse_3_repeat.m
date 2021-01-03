function opt_sat_pulse_3_repeat

% Simulation parameters
numseg=50; %lseg=0.1*(pi/2);
numpts=8192;  maxoffs=20;

params.del_w=linspace(-maxoffs,maxoffs,numpts);
params.wtfun=hamming(numpts)';

count=1;
results={};
while(1)
    params.tp=pi*rand(1,numseg);
    params.phi=2*pi*rand(1,numseg);
    [out]=opt_sat_pulse_3(params);
    
    results{count}=out;
    save results_sat_pulse_3.mat results
    
    disp(out.sat)
    disp(count)
    count=count+1;
end