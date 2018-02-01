close all
clear all
clc
s=tf('s');
options = optimoptions('lsqnonlin');
options.Algorithm = 'trust-region-reflective';
options=optimset('Display','iter');
warning('off', 'Control:analysis:MarginUnstable');

h = 1;  % Time gap
std=2;  % Standstill distance
a_w=2;  % Maxima aceleracion global de comfort

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) Identificar el control de bajo nivel a partir de los vectores entrada
% salida

load('RealData.mat');

G_Veh = tf([8.599], [1 3.231 8.564]);
BW = bandwidth(G_Veh);
margin(G_Veh)

% El algoritmo de optimizacion recibe la funcion donde se calcula el score
% de cada controlador, y minimiza la funcion de coste. El parametro h es el
% time gap del string, C_0 es la semilla de la optimizacion

C_0=[1 1];
func = @(ControlParameters) ControllerOptimization(ControlParameters,h,G_Veh);
[ControllerParameters,val]=fsolve(func,C_0);

% 3) Verificar si la solucion de ControlParameters cumple con los
% requisitos postulados antes

Kp=ControllerParameters(1);
Wc=ControllerParameters(2);
H = 1+h*s;
Gc = Kp*(1 + s/Wc);
% Gc = Kp*(1+100/(1+100/s));

Gpf = G_Veh/(s*(1-G_Veh));

% 2) Una vez obtenido el modelo, dise�ar un controlador que ofrezca el
% mismo ancho de banda que el sistema G_veh, margen de fase de 60� y String
% Stability

[~, Mf_new, ~, Wcg_new] = margin(Gpf*Gc*H);
figure;
margin(Gpf*Gc*H)
% BW_new = bandwidth(Gpf*Gc*H);

% string stability
T = (Gpf*Gc)/(1+Gpf*Gc*H);
[mag, ~] = bode(T);
T_inf = max(mag); %  se puede comprobar que cumple con lael String Variablity

% 4) Probar el controlador encontrado en el modelo de simulink

% Curvature

sim('ACC');

% sim('ACC_Solucion');
