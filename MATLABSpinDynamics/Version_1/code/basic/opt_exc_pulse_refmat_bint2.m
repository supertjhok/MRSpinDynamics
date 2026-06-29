% Optimize CPMG excitation pulse, precalculate refocusing matrix
% Convert into a discrete problem (Gray coded)
% Soumyajit Mandal, 09/22/10

function [texc,pexc,echo_pk,echo_rms]=opt_exc_pulse_refmat_bint2(nseg,T_90,T_FP,refmat,delt)

T_min=0.1*T_90; T_max=2*T_90; % Minimum and maximum segment length
del_ph=pi/2; % Phase quantization step

n1=ceil(log2(T_max/T_min)); % Number of bits to define segment length
n2=ceil(log2(2*pi/del_ph)); % Number of bits to define phase
nbits=nseg*(n1+n2); % Total number of bits

% Excitation pulse definition

% Random initial conditions - duration and phase
% Binary vector guarantees that all segment lengths are positive, no need
% to explicitly specify upper and lower bounds
% MutationFcn = {@mutationuniform,0.05} | @mutationadaptfeasible| {@mutationgaussian}
% FitnessScalingFcn =  @fitscalingshiftlinear | @fitscalingprop | @fitscalingtop  | {@fitscalingrank}
% CrossoverFcn = @crossoverheuristic | {@crossoverscattered} | @crossoverintermediate | @crossoversinglepoint | @crossovertwopoint | @crossoverarithmetic

options=gaoptimset('PopulationType','bitstring','PopulationSize',20,'Display','iter','Generations', 300,'PlotFcns',{@gaplotbestf,@gaplotbestindiv},'TolFun',3e-8,'CrossoverFcn',{@crossoversinglepoint},'MutationFcn',{@mutationuniform,0.02},'FitnessScalingFcn',{@fitscalingrank},'Vectorized','on');

soln=ga(@(params)fit_function(params,T_90,T_FP,refmat,delt,T_min,del_ph,n1,n2),nbits,[],[],[],[],[],[],[],options);

texc=zeros(1,nseg); pexc=texc;
for j=1:nseg
    texc(j)=T_min*gc2dec(soln((j-1)*(n1+n2)+1:(j-1)*(n1+n2)+n1));
    pexc(j)=del_ph*gc2dec(soln((j-1)*(n1+n2)+n1+1:j*(n1+n2)));
end

outs=zeros(1,2);
[outs(1) outs(2)]=cpmg_van_spin_dynamics_refmat(texc,pexc,T_90,T_FP,refmat,delt);
echo_pk=outs(1);
echo_rms=outs(2);

function val=fit_function(params,T_90,T_FP,refmat,delt,T_min,del_ph,n1,n2)

siz=size(params);
pop=siz(1);
nseg=siz(2)/(n1+n2);

val=zeros(pop,1);
for k=1:pop
	texc=zeros(1,nseg); pexc=texc;
	for j=1:nseg
	    texc(j)=T_min*gc2dec(params(k,(j-1)*(n1+n2)+1:(j-1)*(n1+n2)+n1));
	    pexc(j)=del_ph*gc2dec(params(k,(j-1)*(n1+n2)+n1+1:j*(n1+n2)));
	end

	outs=zeros(1,2);
	[outs(1) outs(2)]=cpmg_van_spin_dynamics_refmat(texc,pexc,T_90,T_FP,refmat,delt);

	%val(k)=-outs(1); %Optimize peak
	%val(k)=-outs(2)*0.3e8;  % Optimize RMS
	val(k)=-0.5*(outs(1)+outs(2)*0.3e8); % Optimize both peak and RMS
end
