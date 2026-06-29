% Generic function to plot time-domain echo from magnetization spectrum
% using the IFFT
% ---------------------------------------------------------------------
% zf -> zero filling ratio (to get smoother time domain waveforms)
% df -> frequency offset for receiver

function [echo,tvect]=plot_echo_arb(macq,del_w,df,len_acq,zf)

num=length(del_w);
num_zf=zf*num;

max_w=max(del_w);

necho=2*round(zf*max_w*len_acq/(2*pi)); % Number of points per echo

tmp=zeros(1,num_zf);
tmp((num_zf-num)/2+1:(num_zf+num)/2)=macq;
tmp=fftshift(ifft(fftshift(tmp)));

echo=zf*tmp((num_zf-necho)/2+1:(num_zf+necho)/2);

kvect=-1i*2*pi*linspace(-necho/2,necho/2-1,necho)/(2*max_w*zf);
echo=echo.*exp(df*kvect); % Shift to right receiver frequency

tvect=linspace(-len_acq/2,len_acq/2,necho);