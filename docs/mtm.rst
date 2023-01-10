Multitaper Misfit
=================

.. warning::

    Please refer to [Laske1996]_ for a more rigorous mathematical
    derivation of this misfit function. This documentation page only serves to
    summarize the math for the purpose of explaining the underlying code.

The misfit :math:`\chi_P(\mathbf{m})` measures frequency-dependent phase
differences estimated with multitaper methods. For a given Earth model
:math:`\mathbf{m}`and a single receiver, :math:`\chi_P(\mathbf{m})` is given by

.. math::

    \chi_P (\mathbf{m}) = \frac{1}{2} \int_0^W  W_P(w) \left|
    \frac{ \tau^{\mathbf{d}}(w) - \tau^{\mathbf{s}}(w, \mathbf{m})}
    {\sigma_P(w)} \right|^ 2 dw

:math:`\tau^\mathbf{d}(w)` is the frequency-dependent
phase measurement of the observed data;
:math:`\tau^\mathbf{s}(w, \mathbf{m})` the frequency-dependent
phase measurement of the synthetic data.
The function :math:`W_P(w)` denotes frequency-domain
taper corresponding to the frequency range over which
the measurements are assumed reliable.
:math:`\sigma_P(w)` is associated with the
traveltime uncertainty introduced in making measurements,
which can be estimated with cross-correlation method,
or Jackknife multitaper approach.

The adjoint source for the same receiver is given by

.. math::

    f_P^{\dagger}(t) = \sum_k h_k(t)P_j(t)

in which :math:`h_k(t)` is one (the :math:`k` th) of multi-tapers.

.. math::

    P_j(t) = 2\pi W_p(t) * \Delta \tau(t) * p_j(t) \\
    P_j(w) = 2\pi W_p(w) \Delta \tau(w) * p_j(w)   \\
    p_j(w) = \frac{iw s_j}{\sum_k(iw s_k)(iw s_k)^*} \\
    \Delta \tau(w) = \tau^{\mathbf{d}}(w) - \tau^{\mathbf{s}}(w, \mathbf{m})


Usage
`````

The following code snippets illustrates the basic usage of the multitaper
misfit function.

.. code:: python

    import pyadjoint

    obs, syn = pyadjoint.get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]

    config = pyadjoint.get_config(adjsrc_type="multitaper_misfit", min_period=20.,
                                  max_period=100., taper_percentage=0.3,
                                  taper_type="cos")

    adj_src = pyadjoint.calculate_adjoint_source(config=config,
                                                 observed=obs, synthetic=syn,
                                                 windows=[(800., 900.)]
                                                 )

