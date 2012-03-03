# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:02:39 2012

@author: martin
"""


from geo_field import GeoField


class SurrGeoFieldAR(GeoField):
    """Geo field data class that can construct AR(k) models of each time series
       and construct surrogates of itself from the AR(k) models."""
    
    
    def __init__(self):
        """"""
        GeoField.__init__(self)
        
        
    def prepare_surrogates(self):
        """Prepare fast generation of surrogates."""
        pass
    

    def construct_surrogate(self):
        """Construct a surrogate in-place."""
        pass
