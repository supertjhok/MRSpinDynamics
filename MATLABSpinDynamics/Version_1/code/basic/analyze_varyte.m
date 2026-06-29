function analyze_varyte

% Parameters
ne=32; % Number of echoes
grad_max=20; % G/cm
inits=[4,10,13,16]; % Beginning of experiment numbers
nignore=4; % Ignore first few echoes to avoid transients
T2=110; % ms

gamma=2*pi*4.257; % (krad/sec)/G
D=1.97e-8; % cm^2/ms for water, room temperature

ngrad=length(inits);
for i=1:ngrad
    [data1,parameter1]=readbrukerfile('cpmg_oneshot_sp_tevar',inits(i)); % Rectangular
    [data2,parameter2]=readbrukerfile('cpmg_oneshot_sp_tevar',inits(i)+1); % CP-M8
    [data3,parameter3]=readbrukerfile('cpmg_oneshot_sp_tevar',inits(i)+2); % CP-M15
    
    tmp=parameter1.gp; % All gradients as percentages
    grad=grad_max*tmp(1,3)/1e2; % Gz (G/cm)
    
    dw=parameter1.dw;
    delays=parameter1.delays;
    pulses=parameter1.pulses;
    ns=parameter1.ns; % Number of averages
    
    delt=2*delays(26); % T_FP increment
    T_180=pulses(3)/1e6;
    
    siz=size(data1);
    len=siz(1); % Total length
    le=len/ne; % Samples per echo
    te_vect=dw*linspace(0,le-1,le);
    te_vect=te_vect-max(te_vect)/2;
    
    numstep=siz(2);
    
    T_FP=2*delays(21)+linspace(0,(numstep-1)*delt,numstep);
    T_E=1e3*(T_FP+T_180); % in ms
    
    % Find echo amplitudes (normalized for gradient and number of averages)
    echo1=zeros(ne-nignore,numstep); echo2=echo1; echo3=echo1;
    eta=zeros(ne-nignore,numstep); relax=zeros(ne-nignore,numstep);
    for j=nignore+1:ne
        eta(j-nignore,:)=D*(gamma*grad)^2*T_E.^3*j/12;
        relax(j-nignore,:)=T_E*j;
        echo1(j-nignore,:)=sqrt(trapz(te_vect,abs(data1((j-1)*le+1:j*le,:).^2))).*exp(relax(j-nignore,:)/T2)*grad/ns;
        echo2(j-nignore,:)=sqrt(trapz(te_vect,abs(data2((j-1)*le+1:j*le,:).^2))).*exp(relax(j-nignore,:)/T2)*grad/ns;
        echo3(j-nignore,:)=sqrt(trapz(te_vect,abs(data3((j-1)*le+1:j*le,:).^2))).*exp(relax(j-nignore,:)/T2)*grad/ns;
    end
    
    figure(1); %clf;
    for j=1:numstep
        semilogy(eta(:,j),echo1(:,j),'b-'); hold on;
        semilogy(eta(:,j),echo2(:,j),'r-');
        semilogy(eta(:,j),echo3(:,j),'k-');
    end
    input('Press any key to continue');
end