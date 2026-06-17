% Repeated optimization runs, store current maximum
function opt_exc_pulse_refmat_spe_repeat(nseg,tref,pref,T_90,NE,T_FP,tpar)

count=1;
refmat=calc_refocusing_mat(tref,pref,T_90,NE,T_FP);

results=[];
while(1)
    [texc,pexc,T_ER,spe_rms,echo_rms]=opt_exc_pulse_refmat_spe(nseg,T_90,T_FP,refmat,tpar,[]);
    if count==1
        spe_rms_max=spe_rms;
        echo_rms_max=echo_rms;
        save exc_pulse_results_spe.mat texc pexc T_ER spe_rms echo_rms
        save exc_pulse_results_echo.mat texc pexc T_ER spe_rms echo_rms
    else
        if spe_rms>spe_rms_max
            spe_rms_max=spe_rms;
            save exc_pulse_results_spe.mat texc pexc T_ER spe_rms echo_rms
        end
        if echo_rms>echo_rms_max
            echo_rms_max=echo_rms;
            save exc_pulse_results_echo.mat texc pexc T_ER spe_rms echo_rms
        end
    end
    
    results(count,1:nseg)=texc;
    results(count,nseg+1:2*nseg)=pexc;
    results(count,2*nseg+1)=T_ER;
    results(count,2*nseg+2)=spe_rms;
    results(count,2*nseg+3)=echo_rms;
    save results.mat results
    
    disp(count)
    disp(spe_rms_max)
    disp(echo_rms_max)
    count=count+1;
end
