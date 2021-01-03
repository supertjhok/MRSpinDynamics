function opt_sat_pulse_1_repeat_reopt

% Simulation parameters
numpts=8192;  maxoffs=20;
params.del_w=linspace(-maxoffs,maxoffs,numpts);

% Frequency-domain window function
%params.wtfun=hamming(numpts)'; % Hamming window

params.wtfun=zeros(1,numpts); % Rectangular window
params.wtfun(abs(params.del_w)<maxoffs/4)=1;

tmp=load('results_sat_pulse_1_rectangle.mat'); results_orig=tmp.results;

count=5;
results={};
while(1)
    params.phi=results_orig{count}.phi;
    params.tp=results_orig{count}.tp;
    [out]=opt_sat_pulse_1(params);
    
    results{count}=out;
    save results_sat_pulse_1_rectangle_reoptx.mat results
    
    disp(out.sat)
    disp(count)
    count=count+1;
end