function [eint1,eint2]=plot_envcomp(filname,exptnums,ne,T2)

[tmp,~] = readbrukerfile(filname,exptnums(1)); % Default
sizdat=size(tmp);
npts=sizdat(1)/ne; % Points per echo
data1=tmp(:,sizdat(2)); % Use only data from the last run

[tmp,parameter] = readbrukerfile(filname,exptnums(2)); % With envelope compensation
data2=tmp(:,sizdat(2)); % Use only data from the last run

dw=parameter.dw;
td=parameter.delays;
tp=parameter.pulses;
te=(2*td(21)+tp(3)/1e6); % echo time, sec

tvect=zeros(1,sizdat(1));
echoes1=zeros(npts,ne); echoes2=echoes1;
spectra1=zeros(npts,ne); spectra2=spectra1;

techo=dw*linspace(-npts/2,npts/2-1,npts)';
del_w=linspace(-1/(2*dw),1/(2*dw),npts);

for n=1:ne
    tvect((n-1)*npts+1:n*npts)=n*te+techo; % sec
    echoes1(:,n)=data1((n-1)*npts+1:n*npts)*exp(n*te/T2); % Correct for relaxation
    echoes2(:,n)=data2((n-1)*npts+1:n*npts)*exp(n*te/T2);
    
    spectra1(:,n)=abs(fftshift(fft(real(echoes1(:,n)))+fft(imag(echoes1(:,n)))));
    spectra2(:,n)=abs(fftshift(fft(real(echoes2(:,n)))+fft(imag(echoes2(:,n)))));
end

%echo_asy=echoes1(:,ne); % Asymptotic echo = last echo of default case
echo_asy=echoes2(:,ne); % Asymptotic echo = last echo of actual case
%norm_asy=sqrt(trapz(techo,abs(echo_asy.*echo_asy)));

%spectra_asy=spectra1(:,ne); % Asymptotic spectrum = last spectrum of default case
spectra_asy=spectra2(:,ne); % Asymptotic spectrum = last spectrum of actual case
norm_asy=sqrt(trapz(del_w,spectra_asy.*spectra_asy));

% Matched filtering with asymptotic echo
eint1=zeros(1,ne); eint2=eint1;
for n=1:ne
    %Time domain
    %eint1(n)=trapz(techo,abs(echo_asy.*echoes1(:,n)))/norm_asy;
    %eint2(n)=trapz(techo,abs(echo_asy.*echoes2(:,n)))/norm_asy;
    
    % Frequency domain
    eint1(n)=trapz(del_w,spectra_asy.*spectra1(:,n))/norm_asy;
    eint2(n)=trapz(del_w,spectra_asy.*spectra2(:,n))/norm_asy;
end

figure(1);
plot(tvect*1e3,abs(data1),'r--'); hold on;
plot(tvect*1e3,abs(data2),'b-');
xlabel('Time (ms)');
ylabel('|Echoes|');

figure(2);
plot(techo*1e6,abs(echoes1(:,ne)),'r--'); hold on;
plot(techo*1e6,abs(echoes2(:,ne)),'b-');
xlabel('Time (\mus)');
ylabel('|Asymptotic echo|');

figure(5);
%plot(eint1/norm_asy,'r*-'); hold on; % Normalize echo integrals to 1
%plot(eint2/norm_asy,'bo-');
plot(eint1,'r*-'); hold on; % Don't normalize echo integrals to 1
plot(eint2,'bo-');
xlabel('Echo number');
ylabel('Normalized echo amplitude');