function [outputArg1,outputArg2] = extractAmps(S,s1,s2)
%EXTRACTAMPS Summary of this function goes here
%   S: echo signal
%   s1: calculated s1 component
%   s2: calculated s2 component

s1S = trapz(conj(s1)*S);
s2s2 = trapz(conj(s2)*s2);
s1s2 = trapz(conj(s1)*s2);
s2S = trapz(conj(s2)*S);
s1s1 = trapz(conj(s1)*s1);
s2s1 = trapz(conj(s2)*s1);
s1r = trapz(conj(s1)*r);

a1 = ((s1S*s2s2)-(s1s2*s2S))/(s1s1*s2s2-s1s2*s2s1);
a2 = (s2S*s1s1-s2s1*s1S)/(s1s1*s2s2-s1s2*s2s1*s2s1);

end

