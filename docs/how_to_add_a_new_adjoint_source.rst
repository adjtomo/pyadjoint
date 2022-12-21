How to Add a New Adjoint Source Type
=====================================

A large part of ``Pyadjoint``'s architecture is due to the desire to make it
as easy as possible to add new types of measurements and adjoint sources in
a stable and clean fashion.

To add a new type of adjoint source just copy the
``src/pyadjoint/adjoint_source_types/waveform_misfit.py`` file to a new
file in the same folder. View the most recent version of the file
`here <https://github.com/krischer/pyadjoint/blob/master/src/pyadjoint/adjoint_source_types/waveform_misfit.py>`_.

This particular file has been written in a very verbose way to double as a
template and tutorial for new adjoint source types. The new adjoint source
is defined by 5 things:

**filename**
    The filename with the ``.py`` stripped will be the name of the adjoint
    source used in the
    :func:`~pyadjoint.adjoint_source.calculate_adjoint_source` function.
    Keep it lowercase and use underscores to delimit multiple words.

``VERBOSE_NAME`` variable in the file
    This determines the verbose and pretty name of the adjoint source and
    misfit. This is used for plots and string representations.

``DESCRIPTION`` variable in the file
    Long and detailed description of the misfit and adjoint source including
    formulas, citations, use cases, ... This will be used in the
    auto-generated documentation of the adjoint source and should be as
    detailed as necessary.

``ADDITIONAL_PARAMETERS`` variable in the file, optional
    If the particular adjoint requires additional parameters in addition to
    the default ones, please document them here.


``calculate_adjoint_source()`` function in the file

    Function that will be called when actually calculation the misfit and
    adjoint source. It must a function with the following signature:

    .. code-block:: python

        def calculate_adjoint_source(
                observed, synthetic, min_period, max_period,
                left_window_border, right_window_border,
                adjoint_src, figure, **kwargs):
            pass

    A couple of things to keep in mind:

    1. Calculate the adjoint source if and only if ``adjoint_src`` is ``True``.
    2. Create a plot if and only if ``figure`` is not ``None``.
    3. Don't ``plt.show()`` the plot. This is handled by a higher-level
       function.
    4. Always calculate the misfit value.
    5. The function is responsible for tapering the data and cutting the
       window.
    6. The final adjoint source must have the same number of samples and
       sampling rate as the input data.
    7. Return the already time-reversed adjoint source.
    8. Return a dictionary with the following structure.

        .. code-block:: python

            return {
                "misfit": 1.234,
                "adjoint_source": np.array(..., dtype=np.float64)
            }

        The ``"adjoint_source"`` key-value pair is optional and depends on
        the aforementioned ``adjoint_src`` parameter. Make sure it returns a
        ``float64`` numpy array.



This is all you need to do. ``Pyadjoint`` will discover it and connect with
the rest of its architecture. It will generate documentation and perform a
few basic tests fully automatically. At one point you might want to add
additional test to the new adjoint source.