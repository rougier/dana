#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#
# DANA is a computing framework for the simulation of distributed,
# asynchronous, numerical and adaptive models.
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL: http://www.cecill.info.
#
# As a counterpart to the access to the source code and rights to copy, modify
# and redistribute granted by the license, users are provided only with a
# limited warranty and the software's author, the holder of the economic
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with
# loading, using, modifying and/or developing or reproducing the software by
# the user in light of its specific status of free software, that may mean that
# it is complicated to manipulate, and that also therefore means that it is
# reserved for developers and experienced professionals having in-depth
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling
# the security of their systems and/or data to be ensured and, more generally,
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# -----------------------------------------------------------------------------
#
# References:
# -----------
#
# * M. Guthrie, A Leblois, A. Garenne, T. Boraud "Why is the Striatum Silent? A
#   Computational Model of Multi- modality Decision Making"
#
# -----------------------------------------------------------------------------
from dana import *
from functools import partial
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def format_coord(Z, x, y):
    ''' '''
    if x is None or y is None or Z is None:
        return ''
    x,y = int(x), int(y)
    if 0 <= x < Z.shape[1] and 0 <= y < Z.shape[0]:
        return '[%d,%d]: %s' % (x,y, Z[y,x])
    return ''

def button_press_event(event):
    axis = event.inaxes
    if axis:
        im = axis.im
        group = axis.group
        x,y = int(event.xdata), int(event.ydata)
        for ax in axes.values():
            ax.im.set_data(np.zeros(ax.group.shape)*np.NaN)
        for connection in group._connections:
            for ax in axes.values():
                if connection._source is ax.group.base:
                     V = connection[y,x]
                     ax.im.set_data(V)
    else:
        for axis in axes.values():
            axis.im.set_data(axis.group)
    plt.draw()



def plot(group, row, col, xlabel='', ylabel=''):
    ''' '''
    index = row*6+col
    axis = grid[index]
    axis.format_coord = partial(format_coord, group)
    fig.add_axes(axis)

    chessboard = np.ones(np.array(group.shape)*4)
    chessboard[::2,1::2] = 0.75
    chessboard[1::2,::2] = 0.75
    im = axis.imshow(chessboard, cmap=plt.cm.gray, interpolation='nearest',
                     extent=[0,group.shape[1],0,group.shape[0]], vmin=0, vmax=1)
    plt.hold(True)
    im = axis.imshow(group, origin="lower", cmap=cmap, interpolation="nearest",
                     extent=[0,group.shape[1],0,group.shape[0]]) #, vmin=0, vmax=10)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.group = group
    axis.im = im
    if xlabel:
        axis.text( 0.50, 1.25, xlabel, transform = axis.transAxes,
                   horizontalalignment='center', verticalalignment='bottom')
    if ylabel:
        axis.text(-0.25, 0.50, ylabel, transform = axis.transAxes, rotation='vertical',
                   horizontalalignment='right', verticalalignment='center')
    plt.hold(False)
    return axis



def set_trial():
    m1,m2 = np.random.randint(0,4,2)
    while m2 == m1:
        m2 = np.random.randint(4)
    c1,c2 = np.random.randint(0,4,2)
    while c2 == c1:
        c2 = np.random.randint(4)
    Cortex_mot['Iext']   = 0
    Cortex_cog['Iext']   = 0
    Cortex_ass['Iext']   = 0
    Cortex_mot['Iext'][0,m1]  = 7
    Cortex_cog['Iext'][c1,0]  = 7
    Cortex_ass['Iext'][c1,m1] = 7
    Cortex_mot['Iext'][0,m2]  = 7
    Cortex_cog['Iext'][c2,0]  = 7
    Cortex_ass['Iext'][c2,m2] = 7



# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # Parameters
    cmap           = plt.cm.bone

    Vmin           = +  1.0
    Vmax           = + 20.0
    Vh             = + 20.0
    Vc             = +  3.0

    Cortex_tau     = + 1/10.0
    Cortex_h       = -  3.0
    Cortex_noise   = +  0.01
    Striatum_tau   = + 1/10.0
    Striatum_h     = +  0.0
    Striatum_noise = +  0.001
    STN_tau        = + 1/10.0
    STN_h          = - 10.0
    STN_noise      =    0.001
    GPi_tau        = + 1/10.0
    GPi_h          = + 10.0
    GPi_noise      =    0.03
    Thalamus_tau   = + 1/10.0
    Thalamus_h     = - 40.0
    Thalamus_noise = +  0.001

    def sigmoid(V):
        return Vmin * (Vmax-Vmin)/(1.0+np.exp((Vh-V)/Vc))

    Cortex_mot   = Group( (1,4), '''dV/dt = Cortex_tau * (-V - Cortex_h + I + Iext )
                                    U = np.maximum(V,0); I; Iext''')
    Cortex_cog   = Group( (4,1), '''dV/dt = Cortex_tau * (-V - Cortex_h + I + Iext)
                                    U = np.maximum(V,0); I; Iext''' )
    Cortex_ass   = Group( (4,4), '''dV/dt = Cortex_tau * (-V - Cortex_h + I + Iext)
                                    U = np.maximum(V,0); I; Iext''' )
    Striatum_mot = Group( (1,4), '''dV/dt = Striatum_tau * (-V - Striatum_h + I)
                                    U = sigmoid(V); I''' )
    Striatum_cog = Group( (4,1), '''dV/dt = Striatum_tau * (-V - Striatum_h + I)
                                    U = sigmoid(V); I''' )
    Striatum_ass = Group( (4,4), '''dV/dt = Striatum_tau * (-V - Striatum_h + I)
                                    U = sigmoid(V); I''' )
    STN_mot      = Group( (1,4), '''dV/dt = STN_tau * (-V - STN_h + I)
                                    U = np.maximum(V,0); I''' )
    STN_cog      = Group( (4,1), '''dV/dt = STN_tau * (-V - STN_h + I)
                                    U = np.maximum(V,0); I''' )
    GPi_mot      = Group( (1,4), '''dV/dt = GPi_tau * (-V - GPi_h + I)
                                    U = np.maximum(V,0); I''' )
    GPi_cog      = Group( (4,1), '''dV/dt = GPi_tau * (-V - GPi_h + I)
                                    U = np.maximum(V,0); I''' )
    Thalamus_mot = Group( (1,4), '''dV/dt = Thalamus_tau * (-V -Thalamus_h + I)
                                    U = np.maximum(V,0); I''' )
    Thalamus_cog = Group( (4,1), '''dV/dt = Thalamus_tau * (-V -Thalamus_h + I)
                                    U = np.maximum(V,0); I''' )

    DenseConnection( Cortex_cog('U'),   Striatum_cog('I'), 1.0)
    DenseConnection( Cortex_mot('U'),   Striatum_mot('I'), 1.0)
    DenseConnection( Cortex_ass('U'),   Striatum_ass('I'), 1.0)
    DenseConnection( Cortex_cog('U'),   Striatum_ass('I'), 0.2)
    DenseConnection( Cortex_mot('U'),   Striatum_ass('I'), 0.2)
    DenseConnection( Cortex_cog('U'),   STN_cog('I'),      1.0)
    DenseConnection( Cortex_mot('U'),   STN_mot('I'),      1.0)
    DenseConnection( Striatum_cog('U'), GPi_cog('I'),      2.0)
    DenseConnection( Striatum_mot('U'), GPi_mot('I'),      2.0)
    DenseConnection( Striatum_ass('U'), GPi_cog('I'),      np.ones((1,9))*2.0)
    DenseConnection( Striatum_ass('U'), GPi_mot('I'),      np.ones((9,1))*2.0)
    DenseConnection( STN_cog('U'),      GPi_cog('I'),      np.ones((9,1))*1.0)
    DenseConnection( STN_mot('U'),      GPi_mot('I'),      np.ones((1,9))*1.0)
    DenseConnection( GPi_cog('U'),      Thalamus_cog('I'), 0.5)
    DenseConnection( GPi_mot('U'),      Thalamus_mot('I'), 0.5)
    DenseConnection( Thalamus_cog('U'), Cortex_cog('I'),   1.0)
    DenseConnection( Thalamus_mot('U'), Cortex_mot('I'),   1.0)
#    DenseConnection( Cortex_cog('U'),   Cortex_cog('I'),   1.0)
#    DenseConnection( Cortex_mot('U'),   Cortex_mot('I'),   1.0)


    @after(clock.tick)
    def add_noise(time):
        Cortex_mot['V'] += np.random.uniform(-Cortex_noise/2.0, +Cortex_noise/2.0, Cortex_mot.shape)
        Cortex_cog['V'] += np.random.uniform(-Cortex_noise/2.0, +Cortex_noise/2.0, Cortex_cog.shape)
        Cortex_ass['V'] += np.random.uniform(-Cortex_noise/2.0, +Cortex_noise/2.0, Cortex_ass.shape)

        Striatum_mot['V'] += np.random.uniform(-0.1,+0.1, Striatum_mot.shape)
        Striatum_cog['V'] += np.random.uniform(-0.1,+0.1, Striatum_cog.shape)
        Striatum_ass['V'] += np.random.uniform(-0.1,+0.1, Striatum_ass.shape)
        STN_mot['V'] += np.random.uniform(-0.1,+0.1, STN_mot.shape)
        STN_cog['V'] += np.random.uniform(-0.1,+0.1, STN_cog.shape)
        GPi_mot['V'] += np.random.uniform(-0.1,+0.1, GPi_mot.shape)
        GPi_cog['V'] += np.random.uniform(-0.1,+0.1, GPi_cog.shape)
        Thalamus_mot['V'] += np.random.uniform(-0.1,+0.1, Thalamus_mot.shape)
        Thalamus_cog['V'] += np.random.uniform(-0.1,+0.1, Thalamus_cog.shape)


    # axes = {}
    # fig = plt.figure(figsize=(12,8))
    # grid = ImageGrid(fig, 111, nrows_ncols = (5,6), axes_pad = 0.1,
    #                  share_all = False, add_all=False )
    # axes['Cortex_mot']   = plot( Cortex_mot('V'),   0, 1, xlabel='Motor cortex')
    # axes['Cortex_cog']   = plot( Cortex_cog('V'),   1, 0, ylabel='Cognitive cortex')
    # axes['Cortex_ass']   = plot( Cortex_ass('V'),   1, 1)
    # axes['Striatum_mot'] = plot( Striatum_mot('U'), 0, 5, xlabel='Motor striatum')
    # axes['Striatum_cog'] = plot( Striatum_cog('U'), 1, 4, ylabel='Cognitive striatum')
    # axes['Striatum_ass'] = plot( Striatum_ass('U'), 1, 5)
    # axes['Thalamus_mot'] = plot( Thalamus_mot('U'), 3, 1, xlabel='Motor thalamus')
    # axes['Thalamus_cog'] = plot( Thalamus_cog('U'), 4, 0, ylabel='Cognitive thalamus')
    # axes['STN_mot']      = plot( STN_mot('U'),      3, 3, xlabel='Motor STN')
    # axes['STN_cog']      = plot( STN_cog('U'),      4, 2, ylabel='Cognitive STN')
    # axes['GPi_mot']      = plot( GPi_mot('U'),      3, 5, xlabel='Motor GPi')
    # axes['GPi_cog']      = plot( GPi_cog('U'),      4, 4, ylabel='Cognitive GPi')

    #from matplotlib import mpl
    #ax = fig.add_axes([0.48, 0.54, 0.03, 0.36])
    #norm = mpl.colors.Normalize(vmin=0, vmax=4)
    #mpl.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm)

#    plt.connect('button_press_event', button_press_event)
#    plt.ion()

    # m1,m2 = np.random.randint(0,4,2)
    # while m2 == m1:
    #     m2 = np.random.randint(4)
    # c1,c2 = np.random.randint(0,4,2)
    # while c2 == c1:
    #     c2 = np.random.randint(4)
    # Cortex_mot['Iext']   = 0
    # Cortex_cog['Iext']   = 0
    # Cortex_ass['Iext']   = 0
    # Cortex_mot['Iext'][0,m1]  = 7
    # Cortex_cog['Iext'][c1,0]  = 7
    # Cortex_ass['Iext'][c1,m1] = 7
    # Cortex_mot['Iext'][0,m2]  = 7
    # Cortex_cog['Iext'][c2,0]  = 7
    # Cortex_ass['Iext'][c2,m2] = 7

    @after(clock.tick)
    def print_cortex(time):
        print 'Time : %.3f ms' % time
        print "Motor cortex    ", Cortex_mot['U'].ravel()
        print "Cognitive cortex", Cortex_cog['U'].ravel()
        print

    #     Cortex_mot['Iext'][0,m1]  = 7
    #     Cortex_cog['Iext'][c1,0]  = 7
    #     Cortex_ass['Iext'][c1,m1] = 7
    #     Cortex_mot['Iext'][0,m2]  = 7
    #     Cortex_cog['Iext'][c2,0]  = 7
    #     Cortex_ass['Iext'][c2,m2] = 7

    # @clock.every(25*millisecond)
    # def update(time):
    #     print time
    #     for axis in axes.values():
    #         axis.im.set_data(axis.group)
    #     plt.draw()
    #     print Cortex_mot['V'].max()

    # set_trial()
    run(time=500*millisecond, dt=1*millisecond)

#    plt.ioff()
#    plt.show()
