% Evaluate performance of OCT pulses with relaxation
% expt = Experiment number
% quant = phase quantization (yes/no)
% pulse_num = pulse number for expt = 5
% rat = T1/T2
% clr = plotting command

function [del_t,T1,T2]=...
    oct_pulses_eval_relax(expt,quant,pulse_num,rat,clr)

numpts=10;
T_90=pi/2;
TE=7*pi;

T2=T_90*logspace(1,3,numpts);
T1=T2*rat;

NE_max=20;

% Reference (no relaxation)
[masy_ref,del_w,~]=...
    opt_exc_pulse_mag_results_relax2(expt,quant,pulse_num,NE_max,1e9,1e9);
echo_rms_ref=sqrt(trapz(del_w,abs(masy_ref.^2)));

echo_rms=zeros(1,numpts); tot_time=echo_rms;
for j=1:numpts
    NE=ceil(2.303*T2(j)/TE); % Order of magnitude attenuation
    if NE>NE_max
        NE=NE_max;
    end
    [masy,~,tot_time(j)]=...
        opt_exc_pulse_mag_results_relax2(expt,quant,pulse_num,NE,T1(j),T2(j));
    echo_rms(j)=trapz(del_w,abs(masy_ref.*masy))/echo_rms_ref; % Matched filtering
    
    disp(j)
end

% Estimate amount of relaxation
del_t=tot_time-T2.*log(echo_rms_ref./echo_rms);

figure(2);
semilogx(T2/pi,del_t/pi,clr); hold on;
xlabel('T_{2} / T_{180}');
ylabel('T_{begin} / T_{180}');