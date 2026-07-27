Installation
============

Prerequisites
-------------

The following software must be installed before setting up **cemd**:

* **Python 3.11** — required Python version.
* **Conda** — package and environment manager (`Miniconda or Anaconda <https://www.anaconda.com/docs/main>`__).
* **Packmol** — required for automated system construction (`download <https://m3g.github.io/packmol/>`__). Must be accessible in your ``$PATH``.
* **VMD** — required for system visualization (`download <https://www.ks.uiuc.edu/Research/vmd/>`__). Must be accessible in your ``$PATH``.

Standard Installation
---------------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/JClaverie/cemd.git
   cd cemd

2. Create the conda environment:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate cemd

3. Install the package in editable mode:

.. code-block:: bash

   pip install -e .

GUI Installation
----------------

To use the graphical interface, use the dedicated environment file instead:

.. code-block:: bash

   conda env create -f environment_gui.yml
   conda activate cemd_ui
   pip install -e ".[gui]"

Verifying the Installation
--------------------------

Run the following to confirm that **cemd** is correctly installed:

.. code-block:: python

   import cemd
   from cemd import AtomicSystem
   print(f"CEMD {cemd.__version__} is ready.")