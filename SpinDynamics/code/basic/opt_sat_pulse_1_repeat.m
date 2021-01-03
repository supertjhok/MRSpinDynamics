function opt_sat_pulse_1_repeat

% Simulation parameters
numseg=50; lseg=0.13*(pi/2);
numpts=8192;  maxoffs=20;

params.tp=lseg*ones(1,numseg);
params.del_w=linspace(-maxoffs,maxoffs,numpts);

% Frequency-domain window function
%params.wtfun=hamming(numpts)'; % Hamming window

params.wtfun=zeros(1,numpts); % Rectangular window
params.wtfun(abs(params.del_w)<maxoffs/4)=1;

count=1;
results={};
while(1)
    params.phi=2*pi*rand(1,numseg);
    [out]=opt_sat_pulse_1(params);
    
    results{count}=out;
    save results_sat_pulse_1_rectangle_50_2.mat results
    
    disp(out.sat)
    disp(count)
    count=count+1;
end