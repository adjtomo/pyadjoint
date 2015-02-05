import obspy
tr = obspy.read()[0]
tr.plot()