import obspy
from pyadjoint.utils import taper_window
tr = obspy.read()[0]
taper_window(tr, 4, 11, taper_percentage=0.10, taper_type="hann")
tr.plot()