function val=maximize_echo(params)

T_90=20;
T_FP=200;
tmp=load('refmat.mat'); refmat=tmp.refmat;
delt=40;

val=fit_function(params,T_90,T_FP,refmat,delt);

% Vectorized function for calculating echo size
function val=fit_function(params,T_90,T_FP,refmat,delt)

siz=size(params);
pop=siz(1);
nseg=siz(2)/2;

val=zeros(pop,1);
for k=1:pop
    texc=params(k,1:nseg);
    pexc=params(k,nseg+1:2*nseg);
    
    outs=zeros(1,2);
    [outs(1) outs(2)]=cpmg_van_spin_dynamics_refmat(texc,pexc,T_90,T_FP,refmat,delt);
    
    %val(k)=-outs(1); %Optimize peak
    %val(k)=-outs(2)*0.3e8;  % Optimize RMS
    val(k)=-0.5*(outs(1)+outs(2)*0.3e8); % Optimize both peak and RMS
end