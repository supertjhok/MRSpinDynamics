function [echo_pk,echo_rms,rel_bw]=coil_effects(name,T_90,del_w1,NE,T_FP,T1,T2,f_RF,Q)

f1=1/(4*T_90*1e-6);
rel_bw=f1./(f_RF./Q);
echo_pk=zeros(1,length(Q)); echo_rms=echo_pk;

tmp=load('exc_timings.mat');
results=tmp.results;

tmp=size(results);
numres=tmp(1);
for i=1:numres
    if strcmpi(name,results{i,1})
        texc=T_90*results{i,2};
        pexc=results{i,3};
    end
end

if strcmpi(name,'rectangular')
    % Rectangular refocusing pulse
    tref=2*T_90;
    pref=pi/2;
else
    % RP2-1.0a refocusing pulse
    tref=2*T_90*[0.14,0.72,0.14];
    pref=pi*[3,1,3]/2;
end

%Reference = rectangular, uniform RF, low-Q
Qmin=1;
[echo,tvect,ref_pk,ref_rms]=cpmg_van_spin_dynamics_plot_delw1_coil...
    (T_90,2*T_90,0,pi/2,T_90,0,NE,T_FP,T1,T2,f_RF,Qmin);

% Variable Q
for j=1:length(Q)
    [echo,tvect,echo_pk(j),echo_rms(j)]=cpmg_van_spin_dynamics_plot_delw1_coil...
        (texc,tref,pexc,pref,T_90,del_w1,NE,T_FP,T1,T2,f_RF,Q(j));
    disp(Q(j))
end

% Normalize results
echo_pk=echo_pk/ref_pk;
echo_rms=echo_rms/ref_rms;

figure(2);
plot(rel_bw,echo_pk,'bo-');
xlabel('Relative bandwidth \omega_{1}/(\omega_{RF}/Q)'); hold on;
ylabel('Maximum absolute value');

figure(3);
plot(rel_bw,echo_rms,'rs-');
xlabel('Relative bandwidth \omega_{1}/(\omega_{RF}/Q)'); hold on;
ylabel('Sqaured integral');