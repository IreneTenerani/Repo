Usage
=====

.. _installation:

Installation
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install lumache

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients



The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

   Raised if the kind is invalid.

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

To get a random number you can use ``Pseudo_triangle.ProbabilityDensityFunction()``:

.. autoclass:: Pseudo_triangle.ProbabilityDensityFunction

To make a test you can use ``Pseudo_triangle.test_triangular()``:
.. autofunction:: Pseudo_triangle.test_triangular()
