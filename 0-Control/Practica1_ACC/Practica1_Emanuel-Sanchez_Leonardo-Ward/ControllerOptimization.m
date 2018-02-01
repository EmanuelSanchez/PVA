function F = ControllerOptimization(x,h,sys)
    s=tf('s');
    Kp = x(1);
    Wc = x(2);
    TimeGap = h;

    % Agregar el modelo identificado
    Gp = sys;
    Gpf = Gp/(s*(1-Gp));
    [~,~,~,Wcg_Veh] = margin(Gp);

    H = 1+TimeGap*s;
    Gc = Kp*(1 + s/Wc);
%     Gc = Kp*(1+100/(1+100/s));

    [~, Mf, ~, Wcg] = margin(Gpf*Gc*H);

    % total transfer function
    T = (Gpf*Gc)/(1+Gpf*Gc*H);
    [mag, ~] = bode(T);
    T_inf = max(mag);

    % Pesos
    W1 = 1;
    W2 = 1;
    W3 = 1000;

    F(1) = W1*(Wcg - Wcg_Veh)^2; % Funcion de coste para el ancho de banda
    F(2) = W2*(Mf - 60)^2 % Funcion de coste para el margen de fase
    F(3) = W3*abs(T_inf - 1) % Funcion de coste para la estabilidad de cadena

end
