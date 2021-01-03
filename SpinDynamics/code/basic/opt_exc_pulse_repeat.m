% Repeated optimization runs, store current maximum
function opt_exc_pulse_repeat(nseg,T_90,NE,T_FP,T1,T2)

count=1;
while(1)
    [texc,pexc,echo_pk]=opt_exc_pulse(nseg,T_90,NE,T_FP,T1,T2);
    if count==1
        echo_pk_max=echo_pk;
        save exc_pulse_results.mat texc pexc echo_pk
    else
        if echo_pk>echo_pk_max
            echo_pk_max=echo_pk;
            save exc_pulse_results.mat texc pexc echo_pk
        end
    end
    disp(count)
    disp(echo_pk_max)
    count=count+1;
end