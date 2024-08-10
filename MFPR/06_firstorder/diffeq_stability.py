#!/usr/bin/env python

"""A variety of methods to solve first order ordinary differential equations.

AUTHOR:
    Jonathan Senning <jonathan.senning@gordon.edu>
    Gordon College
    Based Octave functions written in the spring of 1999
    Python version: March 2008, October 2008
"""

import numpy

#-----------------------------------------------------------------------------

def euler( f, x0, t ):
    """Euler's method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = euler(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """

    n = len( t )
    x = numpy.array( [x0] * n )
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        x[i+1] = x[i] + h * f( x[i], t[i] )

    return x

#-----------------------------------------------------------------------------

def heun( f, x0, t ):
    """Heun's method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = heun(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """
    n = len( t )
    x = numpy.array( [x0] * n )
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        k1 = h * f( x[i], t[i] )
        k2 = h * f( x[i] + k1, t[i+1] )
        x[i+1] = x[i] + ( k1 + k2 ) / 2.0

    return x

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

if __name__ == "__main__":
    from pylab import *

    def f( x, t ):
        return -0.05*x

    a, b = ( 0.0, 100.0 )
    x0 = 1.0
    n=100
    t = numpy.linspace( a, b, n )

    h = 1 # 38 is a nice number, so is 40 ... 
    n=100
    print("h={}".format(h))


    # compute various numerical solutions
    #x_heun = heun( f, x0, t )

    # compute true solution values in equal spaced and unequally spaced cases
    x_true = exp(-0.05*t)

    figure(figsize=(10,6))    
    plot( t, x_true, 'g-')
    #for ni in range(10,110,10):
    for ni in range(1,10,1):
        t = numpy.linspace( a, b, ni )
        hi=(b-a)/float(ni)
        x_euler = euler( f, x0, t )
        plot( t, x_euler, '-o',label='h=%.2f'%hi)
    xlabel( '$x$' )
    ylabel( '$y$' )
    title( 'Resitve of $dy/dx = -0.05\, y $, $y(0)=1$' )
    legend()
    show()
