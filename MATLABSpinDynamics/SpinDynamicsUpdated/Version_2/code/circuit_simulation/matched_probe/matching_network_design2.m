% Design of 2 capacitor matching network
% Match coil of inductance L and quality factor Q
% to a real resistance R0 at frequency f0
% ------------------------------------------------------
% This version uses fmincon to ensure C1 and C2 are positive
% ------------------------------------------------------
% Written by: Soumyajit Mandal, 03/28/19

function [C1,C2]=matching_network_design2(L,Q,f0,R0,plt)

w0=2*pi*f0;
Rs=w0*L/Q;
disp(['Coil series resistance = ' num2str(Rs) ' Ohms'])

C0(1)=(w0*R0)/(L*w0^2);
C0(2)=C0(1)*sqrt(Rs/R0);

lb=[0,0];
fun=@(params)optfun(params,L,Rs,w0,R0);

%options = optimset('TolFun',1e-8,'Display','iter','MaxIter',1000);
options = optimset('TolFun',1e-8,'MaxIter',1000);
soln=fmincon(fun,C0,[],[],[],[],lb,[],[],options);

C1=soln(1)/(w0*R0);
C2=soln(2)/(w0*R0);

w=linspace(w0/sqrt(2),sqrt(2)*w0,1e3);
if plt
    plotfun(C1,C2,L,Rs,w,R0);
end

function plotfun(C1,C2,L,Rs,w,R0)

i=sqrt(-1); s=i*w;
zin=(s*L+Rs)./((s*L+Rs).*s*C1+1)+1./(s*C2);
gamma=(zin-R0)./(zin+R0);

figure;
subplot(3,1,1);
plot(w/(2*pi*1e6),real(zin),'r-');
ylabel('Re(Zin), Ohms'); set(gca,'FontSize',14);
subplot(3,1,2);
plot(w/(2*pi*1e6),imag(zin),'b');
ylabel('Im(Zin), Ohms'); set(gca,'FontSize',14);
xlabel('Frequency, MHz');
subplot(3,1,3);
plot(w/(2*pi*1e6),20*log10(abs(gamma)),'b');
ylabel('|S_{11}|, dB'); set(gca,'FontSize',14);
xlabel('Frequency, MHz');

figure
plot(w/(2*pi*1e6),real(zin),'r-','LineWidth',2);
ylabel('Re(Zin), Ohms'); set(gca,'FontSize',14);
xlabel('Frequency, MHz');
xlim([6 11])
hold on
plot([8 8],[0 800],'--','LineWidth',2,'Color','k')
plot([6 11], [50 50],'--','LineWidth',2,'Color','k')
xlim([6 11])
ylim([0 800])
% whiteBg
% setSize
% font
% export_fig('F:\Dropbox\Apps\Overleaf\Portable and Autonomous Magnetic Resonance\Figures\optMatchTuneReal.pdf')


figure
plot(w/(2*pi*1e6),imag(zin),'b','LineWidth',2);
ylabel('Im(Zin), Ohms'); set(gca,'FontSize',14);
xlabel('Frequency, MHz');
hold on
plot([8 8],[1000 -1000],'--','LineWidth',2,'Color','k')
plot([6 11], [0 0],'--','LineWidth',2,'Color','k')
xlim([6 11])
ylim([-600 200])
% whiteBg
% setSize
% font
% export_fig('F:\Dropbox\Apps\Overleaf\Portable and Autonomous Magnetic Resonance\Figures\optMatchTuneImag.pdf')
function err=optfun(params,L,Rs,w0,R0)

C1=params(1);
C2=params(2);

L=L*w0/R0; Rs=Rs/R0; w0=1;
s=1i*w0;

zin=(s*L+Rs)/((s*L+Rs)*s*C1+1)+1/(s*C2);
err=abs(zin-1);