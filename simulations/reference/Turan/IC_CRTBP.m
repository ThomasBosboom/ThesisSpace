% Initial conditions and orbital periods

%1:EML2south, 
%2:EML2south, #
%3:EML2north, 
%4:EML2north 
%5:EML1south 
%6:EML1south,
%7:EML1north 
%8:EML1north
%9:Lunar Elliptic #
%10:Lunar Polar-Circular
%11:NRHO EML2south,
%12:NRHO EML1south 
%13:NRHO EML2north, 
%14:NRHO EML1north 
%15:EML2 Lyapunov 
%16:EML1 Lyapunov

IC= [1.158270618905551 0 -0.128995838046545 0 -0.210737450802563 0;
        1.147342612325716,0,-0.151423081776634,0,-0.219954169502041,0;
        1.158270618905551 0 0.128995838046545 0 -0.210737450802563 0;
        1.147342612325716,0,0.151423081776634,0,-0.219954169502041,0;
        0.830671760182013,0,-0.118400000000000,0,0.233429643755068,0;
        0.835466798864386,0,-0.142400000000000,0,0.252742253545471,0;
        0.830671760182013,0,0.118400000000000,0,0.233429643755068,0;
        0.835466798864386,0,0.142400000000000,0,0.252742253545471,0;
        0.985121349979458 0.001476496155141 0.004925468520363 -0.873297306080392 -1.611900486933861 0;
        0.987844349596793 -0.001299584518251  0.014854319015465  -0.902696556573766 0 0;
        1.027410769048437, 0, -0.185600000000000, 0, -0.114716994633036, 0
        0.924521591441505,0,-0.2180,0,0.123341859235788,0;
        1.027410769048437, 0, 0.185600000000000, 0, -0.114716994633036, 0
        0.924521591441505,0,0.2180,0,0.123341859235788,0;
        1.220007038450076,0,0,0,-0.427552341627659,0;
        0.768838925440319,0,0,0,0.481300000000000,0];
%Orbital Periods
tff=[3.256435367156219;
     3.171923137920527;
     3.256435367156219;
     3.171923137920527;
     2.786366428605159;
     2.759354417540697;
     2.786366428605159;
     2.759354417540697;
     0.103902832143679;
     0.103902832143679;
     1.582220073687549;
     1.804972332208660;
     1.582220073687549;
     1.804972332208660;
     4.310537750816349;
     4.331290836217472];
 
 ICnames=["L2 Southern Halo.a"; 
        "L2 Southern Halo.b";
        "L2 Northern Halo.a"; 
        "L2 Northern Halo.b";
        "L1 Southern Halo.a";
        "L1 Southern Halo.b";
        "L1 Northern Halo.a";
        "L1 Northern Halo.b";
        "Lunar Elliptic Orb.";
        "Lunar Polar-Circular Orb.";
        "L2 Southern NRHO"; 
        "L1 Southern NRHO";
        "L2 Northern NRHO";
        "L1 Northern NRHO";
        "L2 Lyapunov Orb.";
        "L1 Lyapunov Orb."];

 %Lagrangian point locations
 L1 = [0.836890207233573; 0; 0];
 L2 = [1.155701641870813; 0; 0];
 L3 = [-1.005064756017800;0; 0];
 L4 = [0.487844349596793; 0.866025403784439; 0];
 L5 = [0.487844349596793;-0.866025403784439; 0];
     