function [T2,asymp_pow]=plot_echoes_bruker_pcycle(filname,exptnums,necho,nignore)

[tmp1,~] = readbrukerfile(filname,exptnums(1));
[tmp2,parameter] = readbrukerfile(filname,exptnums(2));
tmp=tmp1-tmp2;

sizdat=size(tmp);
npts=sizdat(1)/necho; % Points per echo
data=tmp(:,sizdat(2)); % Use only data from the last run

dw=parameter.dw;
td=parameter.delays;
tp=parameter.pulses;
te=1e3*(2*td(21)+tp(3)/1e6); % echo time, ms

tvect=zeros(1,sizdat(1));
echo_rms=zeros(1,necho); techo=echo_rms;
echo_asymp=zeros(npts,1);
for i=1:necho
    tvect((i-1)*npts+1:i*npts)=(i-1)*te+dw*linspace(0,npts,npts)'*1e3; % ms
    echo_rms(i)=sqrt(trapz(abs(data((i-1)*npts+1:i*npts)).^2));
    techo(i)=(i-1)*te;
    if i>nignore
        echo_asymp=echo_asymp+data((i-1)*npts+1:i*npts);
    end
end

asymp_pow=trapz(abs(echo_asymp).^2);

ind=nignore+1:necho;
p=polyfit(techo(ind),log(echo_rms(ind)),1);
log_echo_est=polyval(p,techo(ind));
echo_est=exp(log_echo_est);

T2=-1/p(1);

figure(1);
plot(tvect,abs(data)); hold on;
set(gca,'FontSize',14);
xlabel('Time (ms)');
ylabel('Absolute value of echo');

figure(2);
semilogy(techo,echo_rms,'bo','LineWidth',2); hold on;
semilogy(techo(ind),echo_est,'b--','LineWidth',2);
set(gca,'FontSize',14);
xlabel('Time (ms)');
ylabel('RMS amplitude of echo');
title(['Estimated T_{2} = ' num2str(T2/1e3) ' sec']);

figure(3);
techo=dw*linspace(-npts/2,npts/2-1,npts)';
plot(techo*1e6,abs(echo_asymp),'k-'); hold on;
xlabel('Time (\mus)');
ylabel('Absolute value of asymptotic echo');

figure(4);
zf=8; % zero-filling to get smoother spectra
fs=1/(techo(2)-techo(1));
fvect=linspace(-fs/2,fs/2,length(techo)*zf);

echo_zf=zeros(zf*length(echo_asymp),1);
echo_zf((zf-1)*length(echo_asymp)/2:(zf+1)*length(echo_asymp)/2-1)=echo_asymp;
%spect=zf*abs(fftshift(fft(echo_zf)));
spect=zf*abs(fftshift(fft(real(echo_zf))+fft(imag(echo_zf))));

plot(fvect/1e3,spect,'k-'); hold on;
set(gca,'FontSize',14);
xlabel('Frequency, kHz');
ylabel('Echo spectrum (amplitude)')