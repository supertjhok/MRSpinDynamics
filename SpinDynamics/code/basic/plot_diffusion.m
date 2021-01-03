function [eta,eint]=plot_diffusion(expt_num,T2)

[data,parameter]=readbrukerfile('diffusion\cpmg_oneshot_sp_tevar',expt_num);

% Parameters
ne=32; % Number of echoes
gmax=20; % Maximum gradient (G/cm)
D=2.1e-8; % Diffusion constant (cm^2/ms)
gamma=2*pi*4.257; % (krad/s)/G

dw=parameter.dw;
delays=parameter.delays;
pulses=parameter.pulses;
grad=parameter.gp;

delt=2*delays(26); % T_FP increment
T_180=pulses(3)/1e6;

siz=size(data);
len=siz(1); % Total length
le=len/ne; % Samples per echo
te_vect=dw*linspace(-le/2,le/2-1,le);
del_w=linspace(-1/(2*dw),1/(2*dw),le);

numstep=siz(2);

T_FP=2*delays(21)+linspace(0,(numstep-1)*delt,numstep);
T_E=T_FP+T_180;

g=grad(1,3)*gmax/100; % Gz (G/cm)
eta=zeros(ne,numstep);
echoes={}; spectra={};

for j=1:numstep
    eta(:,j)=(gamma*g)^2*D*(T_E(j)*1e3)^3*linspace(1,ne,ne)/12;
    for n=1:ne
        % Get individual echoes, correct for relaxation
        echoes{n,j}=data((n-1)*le+1:n*le,j)*exp(n*T_E(j)/T2);
        spectra{n,j}=abs(fftshift(fft(real(echoes{n,j}))+fft(imag(echoes{n,j}))));
    end
end

% Define asymptotic echo spectrum
spectra_asy=spectra{10,1};
norm_asy=sqrt(trapz(del_w,spectra_asy.*spectra_asy));

eint=zeros(ne,numstep);
% Matched  filtering with asymptotic echo
for j=1:numstep
    for n=1:ne
        eint(n,j)=trapz(del_w,spectra_asy.*spectra{n,j})/norm_asy;
    end
end
