clear 
close all


t=0:0.01:4*pi;

x=cos(t);

x2=sum((x.^2).*0.01) 

xm=mean((x.^2)) 

[pxx,f]=pwelch(x);

%sum(pxx)

sum(pxx.*f)


[psd, fpsd] = PSD_ocen450(x,1/0.01);

sum(psd.*fpsd)

