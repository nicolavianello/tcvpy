"""
Fast Reciprocating Probe

Written by Nicola Vianello
"""

import os
import numpy as np
import xray
import tcv

class FastRP(object):
    """

    Load the signals from the Fast Reciprocating Probe
    Diagnostic

    """
    

    
    @staticmethod
    def iSfromshot(shot, stroke=1):
        """

        Return the ion saturation current from shot. If the more than
        one stroke is performed than the return. Remember than
        more than one ion saturation current are available

        """
        conn = tcv.shot(shot)
        if stroke == 1:
            iSat = conn.tdi(r'\FPis1_1')
        else:
            iSat = conn.tdi(r'\FPis1_2')
        # eventually combine the two
        # add in the attributes also the area

    @staticmethod
    def vFfromshot(shot):
        
