.. _ff_database:

Force Field Database
====================

.. raw:: html

   <iframe src="../_static/ff_viewer.html" style="width:100%; height:800px; border:none; border-radius:12px;"></iframe>

Programmatic Access
-------------------

You can also access the database programmatically:

.. code-block:: python

   # Load a specific parameter
   assignments = {'O': 'ospc', 'H': 'hspc'}
   system.set_ff_from_database(assignments)
   
   # Or explore interactively
   system.explore_ff_database()