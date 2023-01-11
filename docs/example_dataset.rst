Example Dataset
===============

This document describes the origin of the example data used in Pyadjoint.

.. code::

    import pyadjoint
    obs, syn = pyadjoint.get_example_data()


The example data features two sets of 3D synthetics. One from the Shakemovie
project, the other from a 2-second Instaseis database using the AK135 Earth
model.

We therefore compare the results of a 3D simulation which includes topography,
ellipticity, etc., against a simulation on a 1D background model with a
spherical Earth.

To establish a more practical terminology, the Shakemovie seismograms (3D)
are "observed" data, whereas the Instaseis seismograms (1D) are
considered "synthetics". For all example code snippet in this documentation,
data are compared the 20 to 100 second period band.

Source and Receiver
-------------------

We use an event from the GCMT catalog:

::

   Event name: 201411150231A
   CMT origin time: 2014-11-15T02:31:50.260000Z
   Assumed half duration:  8.2
   Mw = 7.0
   Scalar Moment = 4.71e+19
   Latitude:  1.97
   Longitude: 126.42
   Depth in km: 37.3

   Exponent for moment tensor:  19    units: N-m
            Mrr     Mtt     Mpp     Mrt     Mrp     Mtp
   CMT     3.970  -0.760  -3.210   0.173  -2.220  -1.970

recorded at station ``SY.DBO`` (``SY`` denotes the synthetic data
network from the Shakemovie project):

::

   Latitude: 43.12, Longitude: -123.24, Elevation: 984.0 m


Data Preprocessing
-------------------

Both data and synthetics are processed to have similar spectral content
and to ensure they are sampled at the same points in time.

The processing applied is similar to a typical preprocessing workflow
applied in full waveform inversions. No instrument response removal is performed
as both data samples are synthetic.

The following code block illustrates how the example data were preprocessed.

.. code:: python

    from obspy.signal.invsim import c_sac_taper
    from obspy.core.util.geodetics import gps2DistAzimuth
    
    f2 = 1.0 / max_period
    f3 = 1.0 / min_period
    f1 = 0.8 * f2
    f4 = 1.2 * f3
    pre_filt = (f1, f2, f3, f4)
    
    def process_function(st):
        st.detrend("linear")
        st.detrend("demean")
        st.taper(max_percentage=0.05, type="hann")
    
        # Perform a frequency domain taper a in response removal
        # just without an actual response...
        for tr in st:
            data = tr.data.astype(np.float64)
    
            # smart calculation of nfft dodging large primes
            from obspy.signal.util import _npts2nfft
            nfft = _npts2nfft(len(data))
    
            fy = 1.0 / (tr.stats.delta * 2.0)
            freqs = np.linspace(0, fy, nfft // 2 + 1)
    
            # transform data to Frequency domain and taper
            data = np.fft.rfft(data, n=nfft)
            data *= c_sac_taper(freqs, flimit=pre_filt)
            data[-1] = abs(data[-1]) + 0.0j

            # transform data back into the time domain
            data = np.fft.irfft(data)[0:len(data)]
            # assign processed data and store processing information
            tr.data = data
    
        st.detrend("linear")
        st.detrend("demean")
        st.taper(max_percentage=0.05, type="hann")
    
        st.interpolate(sampling_rate=sampling_rate, starttime=cmt_time,
                       npts=npts)
    
        _, baz, _ = gps2DistAzimuth(station_latitude, station_longitude,
                                    event_latitude, event_longitude)

        # Rotate to the RTZ coordinate system
        components = [tr.stats.channel[-1] for tr in st]
        if "N" in components and "E" in components:
            st.rotate(method="NE->RT", back_azimuth=baz)
    
        return st
