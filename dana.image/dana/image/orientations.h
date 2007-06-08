class NormOp {
public:
    inline void operator()(double& hor,double& vert,double& norm) {
        norm  = (sqrt((hor*hor + vert*vert)/16.0));
    } 
};    

class AngleOp {
public:
    inline void operator()(double& hor,double& vert,double& angle) {
        //TODO : Correction à apporter..
        // Pour des raisons pratiques, pour le calcul des cartes d'orientations,
        // je calcule une carte avec, pour un chaque pixel, un double qui est la valeur de l'angle en radians
        // puis j'applique un filtre sur cette valeur, ce filtre étant centré sur l'angle de la carte à construire
        // (cf les classes suivantes (Angle135Op, Angle90Op, etc.).
        // par contre, quand hor = 0.0, pour éviter la division par 0, je définis l'angle comme 0 radian
        // mais si vert = 0 et hor = 0, par exemple sur une surface uniforme, on aimerait que la valeur du pixel
        // dans toutes les cartes d'orientations soit nulle, ce qui n'est pas le cas ici

        // voir l'intervalle d'angle pour le resultat de atan et attribuer une valeur
        // en dehors de cet intervalle pour les valeurs non valides
        //atan retourne une valeur entre -pi/2 et pi /2
        if(hor == 0.0)
         	angle = -M_PI;
        else
         	angle  = atan(vert/hor);
    } 
}; 

class Angle135Op {
public:
	inline void operator()(double& angle_src, double& angle_dst)
	{
		// We have a source angle, we apply a gaussian function around 135° and save the result in angle_dst
        if(angle_src == -M_PI)
            angle_dst = 0;
        else
            {
                double value = (1.0+cos((angle_src-M_PI_4)))/2.0;
                angle_dst = exp( - (value-1.0)*(value-1.0)/(0.1*0.1));
            }
	}
};

class Angle90Op {
public:
	inline void operator()(double& angle_src, double& angle_dst)
	{
		// We have a source angle, we apply a gaussian function around 90° and save the result in angle_dst
        if(angle_src == -M_PI)
            angle_dst = 0;
        else
            {
                double value = (1.0+cos(angle_src))/2.0;
                angle_dst = exp( - (value-1.0)*(value-1.0)/(0.1*0.1));
            }
	}
};

class Angle45Op {
public:
	inline void operator()(double& angle_src, double& angle_dst)
	{
		// We have a source angle, we apply a gaussian function around 45° and save the result in angle_dst
        if(angle_src == -M_PI)
            angle_dst = 0;
        else
            {
                double value = (1.0+cos((angle_src+M_PI_4)))/2.0;
                angle_dst = exp( - (value-1.0)*(value-1.0)/(0.1*0.1));
            }   
	}
};

class Angle0Op {
public:
	inline void operator()(double& angle_src, double& angle_dst)
	{
		// We have a source angle, we apply a gaussian function around 0° and save the result in angle_dst
        if(angle_src == -M_PI)
            angle_dst = 0;
        else
            {
                double value = (1.0+cos(angle_src-M_PI_2))/2.0;
                angle_dst = exp( - (value-1.0)*(value-1.0)/(0.1*0.1));
            }
	}
};
