

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import basemap


def render_component(m, C, lats, lons, rmax, sym_clims = False, title = None, cmap = None):
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
    if sym_clims:
        imgplt.set_clim(-rmax, rmax)
        
    plt.colorbar(fraction = 0.07, shrink = 0.7, aspect = 15)
    
    if title != None:
        plt.title(title)
        
    return imgplt
        
        
def render_component_single(C, lats, lons, sym_clims = False, fname = None, plt_name = None, cmap = None):
    """
    Render a single component on a plot.
    """
    rmax = max([np.max(C), np.max(-C)])
    
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
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.95)
    plt.subplot(1, 1, 1)
    render_component(m, Cout[lat_ndx, :], lats_s, lons_s, rmax, sym_clims, plt_name, cmap)
    
    if fname:
        plt.savefig(fname)
        
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
        plt.close()
    else:
        return f


def render_component_set(Comps, names, lats, lons, sym_clims = True, fname = None, plt_name = None):
    """
    Render a component set.  Each component is either 
      - a 2D ndarray with layout corresponding to lats/lons and is plotted using Basemap
      - a tuple with the first element being x-values and the second element being y-values, plotted
        using standard plot
    If fname is not None, the plot is saved to file, then cleared.
    """
    P = len(Comps)
    
    # find max/min over all components
    rmax = max([max(np.max(Ci), abs(np.min(Ci))) for Ci in Comps if type(Ci) == np.ndarray])
    
    # fill in "default" name
    if plt_name == None:
        plt_name = 'Component'

    # in case lats are not in ascending order, fix this
    lat_ndx = np.argsort(lats)
    lats_s = lats[lat_ndx]
    
    # each figure is 6 x 3, how do we arange them?
    rows = int(P**0.5 + 1)
    cols = (P + rows - 1) // rows
        
    f = plt.figure(figsize = (cols * 8, rows * 4))
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.95)
    for i in range(P):
        plt.subplot(rows, cols, i+1)
        if type(Comps[i]) == np.ndarray:
            Ci = Comps[i]
            Ci2, lons_s = basemap.shiftgrid(180, Ci, lons)
            lons_s -= 360
            m = Basemap(projection='mill',
                        llcrnrlat=lats_s[0], urcrnrlat=lats_s[-1],
                        llcrnrlon=lons_s[0], urcrnrlon=lons_s[-1],
                        resolution='c')
            render_component(m, Ci2[lat_ndx, :], lats_s, lons_s,
                             rmax, sym_clims, plt_name + ' - ' + names[i])
        else:
            # it's a 1D plot (time series)
            x, y = Comps[i]
            plt.plot(x, y, 'bo-')
            plt.title(plt_name + ' - ' + names[i])
    
    if fname:
        plt.savefig(fname)
        plt.close()
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
    
