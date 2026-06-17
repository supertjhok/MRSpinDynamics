function plot_excitation_spectrum(pulse)

lskip=3; % Lines to skip
fs=10; % Units of T_180

filname=[pulse, '.txt'];
fhandle=fopen(filname,'r');

for i=1:lskip
    tmp=fgetl(fhandle);
end

cnt=1; dat=[];
while ~feof(fhandle)
    curr=fgetl(fhandle);
    if ~isempty(curr)
       dat(cnt,:)=str2num(curr);
       cnt=cnt+1;
    end
end

fclose(fhandle);

i=sqrt(-1);
dat=dat(:,1).*exp(i*dat(:,2)*pi/180); % Complex data
sizdat=size(dat);

dat_rect=zeros(sizdat(1),1); % Rectangular pulse for comparison
numrect=fs/2; ind=round((sizdat(1)-numrect)/2);
dat_rect(ind:ind+numrect-1)=ones(numrect,1);

dat_ns=(randn(sizdat(1),1)+i*randn(sizdat(1),1))/sqrt(2); % Complex noise pulse for comparison
%dat_ns=randn(sizdat(1),1); % Real noise pulse for comparison
tvect=linspace(0,length(dat)/fs,length(dat)); % Time vector

close all;
figure(1);

subplot(2,1,1);
plot(tvect,abs(dat),'b-'); hold on;
plot(tvect,abs(dat_ns),'k-');
ylabel('Magnitude');

subplot(2,1,2);
plot(tvect,phase(dat)*180/pi,'b-'); hold on;
plot(tvect,phase(dat_ns)*180/pi,'k-');
ylabel('Phase (degrees)');
xlabel('Time (units of T_{180})');

% Calculate spectra
spect=fftshift(fft(dat));
spect_rect=fftshift(fft(dat_rect));
spect_ns=fftshift(fft(dat_ns));
fvect=linspace(-fs/2,fs/2,length(dat)); % Frequency vector

figure(2);

subplot(2,1,1);
plot(fvect,abs(spect),'b-'); hold on;
plot(fvect,abs(spect_rect),'r-');
plot(fvect,abs(spect_ns),'k-');
ylabel('Magnitude');

subplot(2,1,2);
plot(fvect,phase(spect)*180/pi,'b-'); hold on;
plot(fvect,phase(spect_rect)*180/pi,'r-');
plot(fvect,phase(spect_ns)*180/pi,'k-');
ylabel('Phase (degrees)');
xlabel('Frequency (units of 1/T_{180})');