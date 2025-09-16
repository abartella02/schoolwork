clc%clears screen
clear all%clears history
close all%closes all files
mu=0.012277471;
mud=1-mu;
f=@(t,y) [y(2),y(1) - 2*y(4) - (mud*y(1))/(y(1)^2 + (mu + y(3))^2)^(3/2) - (mu*y(1))/(y(1)^2 + (mud - y(3))^2)^(3/2),y(4), 2*y(2) + y(3) - (mud*(mu + y(3)))/(y(1)^2 + (mu + y(3))^2)^(3/2) + (mu*(mud - y(3)))/(y(1)^2 + (mud - y(3))^2)^(3/2)];
for N=[100,1000,10000,20000]
    figure;
    [X,Y]=runge4(f,[0,17.1],[0.994,0,0,-2.001585106379082522420537862224],N);
    plot(Y(:,1),Y(:,3),'r', 'LineWidth', 3);
    xlabel('u_1(t)');
    ylabel('u_2(t)');
    title([num2str(N) ' Steps'], FontSize=16);
end

function [x,y]=runge4(f,tspan,y0,N)
    x = linspace(tspan(1),tspan(2),N);
    h=x(2)-x(1);
    % Calculates upto y(3)
    y = zeros(length(x),4);
    y(1,:) = y0; % initial condition
    for i=1:(length(x)-1) % calculation loop
        k_1 = f(x(i),y(i,:));
        k_2 = f(x(i)+0.5*h,y(i,:)+0.5*h*k_1);
        k_3 = f((x(i)+0.5*h),(y(i,:)+0.5*h*k_2));
        k_4 = f((x(i)+h),(y(i,:)+k_3*h));
        y(i+1,:) = y(i,:) + (1/6)*(k_1+2*k_2+2*k_3+k_4)*h; % main equation
    end
end