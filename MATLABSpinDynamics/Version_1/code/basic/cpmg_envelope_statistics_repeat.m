function [results1,results2]=cpmg_envelope_statistics_repeat(ne,T_E,sigma,numruns)

results1={}; results2={};

% No timing noise
for sel=0:3
    [eint0,eint,eint_mean,eint_std]=cpmg_envelope_statistics(sel,ne,T_E,sigma,numruns,0);
    results1{sel+1,1}=eint0;
    results1{sel+1,2}=eint;
    results1{sel+1,3}=eint_mean;
    results1{sel+1,4}=eint_std;
end

% Timing noise
for sel=0:3
    [eint0,eint,eint_mean,eint_std]=cpmg_envelope_statistics(sel,ne,T_E,sigma,numruns,1);
    results2{sel+1,1}=eint0;
    results2{sel+1,2}=eint;
    results2{sel+1,3}=eint_mean;
    results2{sel+1,4}=eint_std;
end