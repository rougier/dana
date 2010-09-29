import math
 
def Runge_Kutta_step(y, t, delta, f):
    k1 = f( y , t)
    k2 = f( y + 0.5 * k1 * delta , t + delta_t / 2)
    k3 = f( y + 0.5 * k2 * delta , t + delta_t / 2 )
    k4 = f( y + k3 * delta , t + delta_t )
    return y + ( k1 + ( 2 * k2 ) + ( 2 * k3 ) + k4 ) * ( delta / 6 )
 
def f(y,t) :
    return math.exp(t)
 
vec_t = list()     # liste des valeurs en t
vec_y = list()     # liste des valeurs en y (inconnue)
t = 0              # valeur de depart pour t
t_max = 10         # valeur finale pour t
y = 1              # condition initiale de l'equation differentielle
delta_t = 1e-3     # pas d'integration
 
while t<t_max :      # la boucle d'integration a proprement dite.
    vec_t.append(t)  # sauvegarde de la valeur courante de t
    vec_y.append(y)  # sauvegarde de la valeur courante de y
    y = Runge_Kutta_step(y, t,delta_t, f)  # incrementation de la valeur de y
    # par la formule de runge kutta definie dans 'Runge_Kutta_step' :
    # formule y_{i+1} = y_{i} + (h/6) ( k1 + 2 k2 + 2 k3 + k4 )
    t = t + delta_t  #nouvelle valeur de t apres incrementation de la solution
 
try :                # trace de la solution 
    from pylab import plot, xlabel, ylabel, show
    plot( vec_t , vec_y )

    xlabel("t")
    ylabel("y(t) = ")
    show()
except ImportError:
    print("Installez matplotlib pour visualiser la solution")
