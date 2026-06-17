% texc,pexc = excitation pulse times, phases
% tref,pref = refocusing pulse times, phases
% NE = number of echoes
% TE = echo spacing
% Include effects of RF inhomogeneity and tuned coil

% Delays follow pulses

function [echo,tvect,echo_pk,echo_rms]=cpmg_van_spin_dynamics_plot_delw1_coil(texc,tref,pexc,pref,T_90,del_w1,NE,T_FP,T1,T2,f_RF,Q)

nexc=length(texc);
nref=length(tref);

tp=zeros(1,nexc+NE*nref);
phi=tp;
tf=tp;

tp(1:nexc)=texc;
phi(1:nexc)=pexc;
tf(1:nexc-1)=zeros(1,nexc-1);

if nexc==1 % Assume rectangular pulse
    tf(nexc)=0.5*T_FP-2*T_90/pi; % Martin's timing correction
else % Not a rectangular pulse
    tf(nexc)=0.5*T_FP;
end

for i=1:NE
    tp(nexc+(i-1)*nref+1:nexc+i*nref)=tref;
    phi(nexc+(i-1)*nref+1:nexc+i*nref)=pref;
    tf(nexc+(i-1)*nref+1:nexc+i*nref-1)=zeros(1,nref-1);
    tf(nexc+i*nref)=T_FP;
end
tf(nexc+NE*nref)=T_FP/2;

[echo,tvect]=sim_spin_dynamics_allpw_delw1_coil(T_90,del_w1,tp,phi,tf,T1,T2,f_RF,Q);
echo_pk=max(abs(echo));
echo_rms=trapz(tvect,abs(echo).^2)*1e8;

figure(1); %clf;
%plot(tvect*1e6,real(echo),'b-'); hold on;
%plot(tvect*1e6,imag(echo),'r-');
plot(tvect*1e6,abs(echo),'k-'); hold on;
set(gca,'FontSize',14);
xlabel('Time (\mus)');
ylabel('Waveform of echo, s(t)')