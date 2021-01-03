% Plot normalized SNR for OCT excitation pulses with variable number of
% segments

function plot_normalized_snr2(datafile,clr)

tmp=load(datafile);
results=tmp.results;
tmp=size(results); nump=tmp(1);

% Rectangular 90/180 CPMG as reference
T_90=pi/2; T_180=2*T_90; % normalized
tref=T_180*[3 1 3]; aref=[0,1,0]; pref=[0,0,0];
texc=T_90; aexc=1; pexc=pi/2;
len_acq=3*T_180;
[neff,del_w]=calc_rot_axis_arba(tref,pref,aref);
[masy]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);

delt=0.01*T_90;
tvect=-len_acq/2:delt:len_acq/2;
echo=zeros(1,length(tvect));
for i=1:length(tvect)
    echo(i)=sum(masy.*exp(-1i*del_w*tvect(i)));
end
rect_rms=sqrt(trapz(tvect,abs(echo).^2));

rms=zeros(1,nump); Nseg=rms;
for i=1:nump
    tmp=results{i,1}; Nseg(i)=length(tmp);
    rms(i)=results{i,4};
end

figure(1);
plot(Nseg,(rms/rect_rms).^2,clr); hold on;
ylabel('Normalized SNR (power units)')
xlabel('Number of segments')