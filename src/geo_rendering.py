

from mpl_toolkits.basemap import Basemap
from mpl_toolkits import basemap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime


def render_component(m, C, lats, lons, clims = None, title = None, cmap = None, cbar = True, cbticks = None):
    """
    Render a single component onto the current axes.  Lats/lons must
    be sorted and must correspond to the 2D component C.  The image
    will be drawn into the given axes. A colorbar is drawn automatically,
    a title is set optionally.
    """
    m.drawcoastlines()
#    m.etopo(scale = 0.2)
    m.drawparallels(np.arange(-90.,91.,30.), labels=[1,0,0,1])
    m.drawmeridians(np.arange(-120., 121.,60.), labels=[1,0,0,1])
   
    nx = int((m.xmax-m.xmin) / 20000) + 1
    ny = int((m.ymax-m.ymin) / 20000) + 1
    f = m.transform_scalar(C, lons, lats, nx, ny)
    
    imgplt = m.imshow(f, alpha = 0.8, cmap = cmap)
    if clims:
        imgplt.set_clim(clims[0], clims[1])
        
    if cbar:
        plt.colorbar(fraction = 0.07, shrink = 0.5, aspect = 15, ticks = cbticks)
    
    if title != None:
        plt.title(title)
        
    return imgplt
        
        
def render_component_single(C, lats, lons, clims = None, fname = None, plt_name = None,
                            cmap = None, cbar = True, cbticks = None):
    """
    Render a single component on a plot.
    """
    
    # we either accept prescribed
    if clims == 'symm':
        rmax = max(np.amax(C), np.amax(-C))
        clims = (-rmax, rmax)
    
    if plt_name == None:
        plt_name = 'Component'

    # in case lats are not in ascending order, fix this
    lat_ndx = np.argsort(lats)
    lats_s = lats[lat_ndx]
    
    # shift the grid by 180 degs and remap lons to -180, 180
    Cout, lons_s = basemap.shiftgrid(180, C, lons)
    lons_s -= 360
        
    # construct the projection from the remapped data
    m = Basemap(projection='mill',
                llcrnrlat=lats_s[0], urcrnrlat=lats_s[-1],
                llcrnrlon=lons_s[0], urcrnrlon=lons_s[-1],
                resolution='c')

    f = plt.figure()
    if cbar:
        plt.subplots_adjust(left = 0.1, bottom = 0.05, right = 0.95, top = 0.95)
    else:
        plt.subplots_adjust(left = 0.1, bottom = 0.05, right = 1.0, top = 1.0)

    plt.subplot(1, 1, 1)
    render_component(m, Cout[lat_ndx, :], lats_s, lons_s, clims, plt_name, cmap, cbar, cbticks)
    
    if fname:
        plt.savefig(fname)
        plt.close(f)
    else:
        return f
    

def render_component_triple(C1, C2, C3, names, lats, lons, sym_clims = True, fname = None, plt_name = None):
    """
    Render a component triple, useful in some contexts.  The names of each subplot must be given as well
    as the latitudes, longitudes, where the plots are to be shown.  Optionally, the color limits of all three
    plots are forced to be equal and symmetric around zero (sym_clims).  If fname is not none, the plots are
    saved to file, otherwise they remain in memory and can be shown at will.
    """
    rmax = max([np.max(C1), abs(np.min(C1)), np.max(C2), abs(np.min(C2)), np.max(C3), abs(np.min(C3))])
    
    if plt_name == None:
        plt_name = 'Component'

    # in case lats are not in ascending order, fix this
    lat_ndx = np.argsort(lats)
    lats_s = lats[lat_ndx]
        
    m = Basemap(projection='mill',
                llcrnrlat=lats_s[0], urcrnrlat=lats_s[-1],
                llcrnrlon=lons[0], urcrnrlon=lons[-1],
                resolution='c')

    f = plt.figure(figsize = (20,20))
    plt.subplot(3, 1, 1)
    render_component(m, C1[lat_ndx, :], lats_s, lons, rmax, sym_clims, plt_name + ' - ' + names[0])
    plt.subplot(3, 1, 2)
    render_component(m, C2[lat_ndx, :], lats_s, lons, rmax, sym_clims, plt_name + ' - ' + names[1])
    plt.subplot(3, 1, 3)
    render_component(m, C3[lat_ndx, :], lats_s, lons, rmax, sym_clims, plt_name + ' - ' + names[2])
    
    if fname:
        plt.savefig(fname)
        plt.close(f)
    else:
        return f


def render_component_set(Comps, names, lats, lons, clims = None, fname = None, plt_name = None):
    """
    Render a component set.  Each component is either 
      - a 2D ndarray with layout corresponding to lats/lons and is plotted using Basemap
      - a tuple with the first element being x-values and the second element being y-values, plotted
        using standard plot
    If fname is not None, the plot is saved to file, then cleared.
    """
    P = len(Comps)
    
    # check required color limits
    if clims == 'symm':
        rmax = max([max(np.max(Ci), abs(np.min(Ci))) for Ci in Comps if type(Ci) == np.ndarray])
        clims = (-rmax, rmax)
    elif clims == 'same':
        rmax = max([np.max(Ci) for Ci in Comps if type(Ci) == np.ndarray])
        rmin = max([np.min(Ci) for Ci in Comps if type(Ci) == np.ndarray])
        clims = (rmin, rmax)
    
    # fill in "default" name
    if plt_name == None:
        plt_name = 'Component'

    # in case lats are not in ascending order, fix this
    lat_ndx = np.argsort(lats)
    lats_s = lats[lat_ndx]
    
    # each figure is 6 x 3, how do we arange them?
    rows = int(P**0.5 + 0.5)
    cols = (P + rows - 1) // rows
        
    f = plt.figure(figsize = (cols * 8, rows * 4))
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.95)
    for i in range(P):
        if i == 2 and P == 3:
            ax = plt.subplot(2, 1, 2)
        else:
            ax = plt.subplot(rows, cols, i+1)
        
        if type(Comps[i]) == np.ndarray:
            Ci = Comps[i]
            Ci2, lons_s = basemap.shiftgrid(180, Ci, lons)
            lons_s -= 360
            m = Basemap(projection='mill',
                        llcrnrlat=lats_s[0], urcrnrlat=lats_s[-1],
                        llcrnrlon=lons_s[0], urcrnrlon=lons_s[-1],
                        resolution='c')
            render_component(m, Ci2[lat_ndx, :], lats_s, lons_s,
                             clims, plt_name + ' - ' + names[i])
        elif type(Comps[i]) == tuple:
            ptype, x, y = Comps[i]
            
            if ptype == 'date': 
                # it's a 1D plot (time series with time)
                d = [datetime.date.fromordinal(int(xi)) for xi in x]
                plt.plot(d, y, 'b-')
                for label in ax.get_xticklabels():
                    label.set_rotation(30)
                    label.set_horizontalalignment('right')
                ax.fmt_xdata = matplotlib.dates.DateFormatter('%Y-%m-%d')         
                plt.title(plt_name + ' - ' + names[i])
            elif ptype == 'freq':
                plt.plot(x, y)
                plt.grid()
                plt.xlabel('Angular frequency [rad/sample]')
                plt.ylabel('Frequency content [dB]')
                plt.title(plt_name + ' - ' + names[i])
            elif ptype == 'invfreq':
                plt.plot(x, y)
                plt.grid()
                plt.xlabel('Period')
                plt.ylabel('Frequency content')
                plt.title(plt_name + ' - ' + names[i])
            elif ptype == 'plot': 
                plt.plot(x, y, 'b-')
                plt.xlabel('Samples')
                plt.title(plt_name + ' - ' + names[i])
            else:
                # leave empty, not understood
                plt.title('Plot data not understood.')
                
    
    if fname:
        plt.savefig(fname)
        plt.close(f)
    else:
        return f



def render_components(C, lats, lons, fname_tmpl = None, ndx = None):
    """
    Render the components in C [with dims comp_id x lats x lons] onto
    a world map of appropriate size.  The method assumes that the  lons x lats
    generate a rectangle.
    """
    m = Basemap(projection='mill',llcrnrlat=min(lats), urcrnrlat=max(lats), llcrnrlon=(min(lons)),
                urcrnrlon=max(lons),resolution='c')
    
    if ndx == None:
        ndx = np.arange(len(C)) + 1
        
    rmax = np.max(C)

    # lattitudes may not be sorted in ascending order, rectify this
    lat_ndx = np.argsort(lats)
    lats_s = lats[lat_ndx]

    # render each topo plot
    for ci in range(len(C)):
        
        plt.figure(figsize=(12, 8 * (max(lats) - min(lats)) / 180))
        plt.axes([0.05, 0.05, 0.9, 0.85])

        m.drawcoastlines()
        #m.fillcontinents(color='coral',lake_color='aqua', zorder = 0)
        m.etopo(scale = 0.2)

        # draw parallels and meridians.
        m.drawparallels(np.arange(-90.,91.,30.))
        m.drawmeridians(np.arange(0.,361.,60.))
        #m.drawmapboundary(fill_color='aqua')
        
        nx = int((m.xmax-m.xmin) / 20000) + 1
        ny = int((m.ymax-m.ymin) / 20000) + 1
        Ci = C[ci, lat_ndx, :]
        f = m.transform_scalar(Ci, lons, lats_s, nx, ny)
        
        # imlim seems to be equivalent to caxis() im MATLAB
        imgplt = m.imshow(f, alpha = 0.8)
        imgplt.set_clim(-rmax, rmax)
        plt.colorbar()
        
        plt.title('Component %d' % (ndx[ci]))
        
        if fname_tmpl:
            plt.savefig(fname_tmpl % (ndx[ci]))
            
    if not fname_tmpl:
        plt.show()
    


def plot_clusters_robinson(ldata, lats, lons, centers = None, nstep = None,
                           clims = None, subplot = False, euro_centered = True, filename = None):
    """ Plot climatic data using robinson projection
    
        Args:
            ldata: Data to be plotted. It should have a shape as len(lats) x 
                len(lons).
            lats: lattitudes (it is assumed that these would be in 90 .. -90)
            lons: longitudes (it is assumed that these would be in 0 .. 360)
            
        Kwargs:
            nsteps:number of steps in values
            clims: Minimum and maximum of corresponding colorbar. If value is 
                'binary' it means to show.
            subplot: Consider plotting as subplot
            euro_centered: horizontal center of plot is on meridian 0 if true, on meridian 180 if false
            filename: if None, plot to buffer, otherwise calls savefig()
    """
    fig = plt.figure(figsize=(15, 8))
    ax = plt.subplot(111)    
    
    # adjust missing last value
    data = np.zeros((ldata.shape[0],ldata.shape[1]+1))
    data[:,:-1] = ldata
    data[:,-1] = data[:,0]
    llons = lons.tolist()
    llons.append(360)
    lons = np.array(llons)

    # automatically compute the number of elements in the data if none set
    if nstep is None:
        nstep = len(np.unique(ldata))
  
    # shift data so lons go from -180 to 180 instead of 0 to 360.
    if euro_centered:
        data,lons = basemap.shiftgrid(180.,data,lons,start=False)
        m = Basemap(projection='robin',lon_0=0,ax=ax)
    else:
        m = Basemap(projection='robin',lon_0=180, ax=ax)
    
    n = ldata.shape[0]
    
    # make filled contour plot.
    x, y = m(*np.meshgrid(lons, lats))
    cmap = matplotlib.cm.jet
    cs = m.pcolor(x,y,data)
    # draw coastlines.
    m.drawcoastlines()
    # draw a line around the map region.
    m.drawmapboundary()
    # draw parallels and meridians.
    m.drawparallels(np.arange(-60.,90.,30.),labels=[1,0,0,0])
    #m.drawmeridians(np.arange(0.,420.,60.),labels=[0,0,0,1])
    m.drawmeridians(np.arange(0.,350.,60.),labels=[0,0,0,1])
    
    # add a colorbar
    cb = m.colorbar(cs,"right", size="2%", pad='2%')
    # apply minimum and maxium
    if (clims == None):
        clims = [data.min(), data.max()]
        
    # setting ticks
    if (clims == 'binary'):
        clims = [data.min(), data.max()]

    # draw the centers black
    if centers is not None:
        for c_i in centers:
            x, y = m(*c_i)
            m.plot(x, y, 'ko')
        
    step = (clims[1] - clims[0]) / float(nstep)
    tcs = [clims[0]]
    tcs.extend([clims[0] + x*step for x in range(1,nstep+1)])
    tcsl = []
    sstep = nstep / 10
    for i in xrange(1,nstep+1):
        if (i % sstep == 0):
            tcsl.append(str(i))
        else:
            tcsl.append('')
    tcs = [int(t) for t in tcs]
    cb.set_ticks(tcs)
    cb.set_ticklabels(tcsl)
    matplotlib.rcParams.update({'font.size': 10})
    
    if filename is not None:
        fig.savefig(filename, bbox_inches = 'tight', pad_inches = 0.1, transparent = False)
    elif (subplot == False):
        plt.show()
    
    return m,lats,lons



def plot_component_robinson(c, ts, lats, lons,
                            cc, exp_var_frac, clims = None, euro_centered = True, filename=None):
    """ Plot a component in standard format with component in top left, center of component marked in top right
        and its corresponding time series shown in bottom half of image.

        Args:
            ldata: Data to be plotted. It should have a shape as len(lats) x 
                len(lons).
            ts: the time series corresponding to the component
            lats: lattitudes (it is assumed that these would be in 90 .. -90)
            lons: longitudes (it is assumed that these would be in 0 .. 360)
            cc: the component center as a tuple (lon, lat)
            clims: optional colormap limits (automatic if None)
            filename: if None, each plot is shown on screen, otherwise savefig() is called
    """

    fig = plt.figure(figsize=(15,8))
    gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[6, 4],height_ratios=[6,4]) 

    plt.subplot(gs[0,:-1])
    plot_data_robinson(c, lats, lons, clims, euro_centered = euro_centered, subplot = True)
    plt.title('Explained variance %g%%' % (round(1000.0 * exp_var_frac)/10.0))
    plt.subplot(gs[0,1])
    m = plot_empty_robinson(lats, lons, euro_centered = euro_centered)
    x,y = m(*cc)
    m.plot(x,y,'bo')
    
    plt.subplot(gs[1,:])
    plt.plot(ts)
    years = range(1949,2010,5)
    sy = [str(y) for y in years]
    plt.xticks(range(12,len(ts),60),sy,rotation=30)
    
    plt.tight_layout()
        
    if filename is not None:
        fig.savefig(filename, bbox_inches = 'tight', pad_inches = 0.5, transparent = False)
    else:
        plt.show()



# Original code of function David Hartman, modified by Martin Vejmelka
def plot_data_robinson(ldata, lats, lons, clims = None, subplot = False, add_colorbar = True,
                       parallel_labels = 'left', euro_centered = True, filename=None):
    """ Plot climatic data using robinson projection
    
        Args:
            ldata: Data to be plotted. It should have a shape as len(lats) x 
                len(lons).
            lats: lattitudes (it is assumed that these would be in 90 .. -90)
            lons: longitudes (it is assumed that these would be in 0 .. 360)
            euro_centered: horizontal center of plot is on meridian 0 if true, on meridian 180 if false
            
        Kwargs:
            clims: Minimum and maximum of corresponding colorbar. If value is 
                'binary' it means to show.
            subplot: Consider plotting as subplot
    """
    # adjust missing last value
    data = np.zeros((ldata.shape[0],ldata.shape[1]+1))
    data[:,:-1] = ldata
    data[:,-1] = data[:,0]
    llons = lons.tolist()
    llons.append(360)
    lons = np.array(llons)
  
    # center on europe (default) or in Pacific
    if euro_centered:
        # shift data so lons go from -180 to 180 instead of 0 to 360.
        data,lons = basemap.shiftgrid(180.,data,lons,start=False)
        # create Basemap instance for Robinson projection.
        m = Basemap(projection = 'robin', lon_0 = 0)
    else:
        m = Basemap(projection = 'robin', lon_0 = 180)
    
    # make filled contour plot.
    x, y = m(*np.meshgrid(lons, lats))
    cmap = matplotlib.cm.jet
    if (clims != None):
        if (clims == 'binary'):
            norm=matplotlib.colors.Normalize(data.min(),data.max()/float(20))
        else:
            norm=matplotlib.colors.Normalize(clim[0],clim[1])

        cs = m.contourf(x,y,data,20,cmap=matplotlib.cm.jet,norm=norm)
    else:
        cs = m.contourf(x,y,data,20,cmap=matplotlib.cm.jet)

    # draw coastlines.
    m.drawcoastlines()
    # draw a line around the map region.
    m.drawmapboundary()
    # draw parallels and meridians.
    if parallel_labels == 'left':
        m.drawparallels(np.arange(-60.,90.,30.),labels=[1,0,0,0])
    else:
        m.drawparallels(np.arange(-60.,90.,30.),labels=[0,1,0,0])
    
    #m.drawmeridians(np.arange(0.,420.,60.),labels=[0,0,0,1])
    if euro_centered:
        m.drawmeridians([-135., -45., 45., 135.], labels=[0,0,0,1])
    else:
        m.drawmeridians([-135., -45., 45., 135.], labels=[0,0,0,1])

    # add a colorbar
    if add_colorbar:
        cb = m.colorbar(cs,"right", size="3%", pad='2%')
        # apply minimum and maxium
        if (clims == None):
            clims = [data.min(), data.max()]
        
        # setting ticks of color bar
        if (clims == 'binary'):
            clims = [data.min(), data.max()]
        nstep = 5
        step = (clims[1] - clims[0]) / float(nstep)
        tcs = [clims[0]]
        tcs.extend([clims[0] + x*step for x in range(1,nstep+1)])
        tcs = [round(t,3) for t in tcs]
        cb.set_ticks(tcs)
        cb.set_ticklabels(tcs)

    if filename is not None:
        plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0.5, transparent = False)
    elif not subplot:
        plt.show()




def plot_empty_robinson(lats, lons, euro_centered = True):
    """ Plot climatic data using robinson projection
    
        Args:
            lats: lattitudes (it is assumed that these would be in 90 .. -90)
            lons: longitudes (it is assumed that these would be in 0 .. 360)
            euro_centered: horizontal center of plot is on meridian 0 if true, on meridian 180 if false
    """
  
    # adjust missing last value
    llons = lons.tolist()
    llons.append(360)
    lons = np.array(llons)
  
    # shift data so lons go from -180 to 180 instead of 0 to 360.
    # create Basemap instance for Robinson projection.
    lon_0 = 0.0 if euro_centered else 180.0
    m = Basemap(projection='robin',lon_0=lon_0)

    # draw coastlines.
    m.drawcoastlines()
    # draw a line around the map region.
    m.drawmapboundary()
    # draw parallels and meridians.
    m.drawparallels(np.arange(-60.,90.,30.),labels=[0,1,0,0])
    #m.drawmeridians(np.arange(0.,420.,60.),labels=[0,0,0,1])
    m.drawmeridians(np.arange(0.,350.,60.),labels=[0,0,0,0])
    #m.fillcontinents(color='coral',lake_color='aqua')
    
    return m
