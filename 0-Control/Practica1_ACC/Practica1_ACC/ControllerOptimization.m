function F = ControllerOptimization(x,h)
    s=tf('s');
    Kp = x(1);
    Wc = x(2);
    TimeGap = h;
    
    % Agregar el modelo identificado
    Gp = ;
    Gpf = Gp/(s*(1-Gp));
    [~,~,~,Wcg_Veh] = margin(Gp);
    
    H = 1+TimeGap*s;
    Gc = Kp*(1 + s/Wc);
    
    F(1) = % Funcion de coste para el ancho de banda
    F(2) = % Funcion de coste para el margen de fase
    F(3) = % Funcion de coste para la estabilidad de cadena

end