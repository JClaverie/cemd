Installation
============

Follow these instructions to set up your development environment and install the **cemd** package.

Prerequisites
-------------
* **Python 3.10+**
* **Conda** (Miniconda or Anaconda)
* **Packmol** (Required for system construction)
* **VMD** (Required for system visualization)

Step-by-Step Installation
-------------------------

1. Clone the repository:
   .. code-block:: bash

      git clone https://github.com/votre-utilisateur/cemd.git
      cd cemd

2. Create the environment from the provided YAML file:
   .. code-block:: bash

      conda env create -f environment.yml
      conda activate cemd_env

3. Install the package in editable mode:
   .. code-block:: bash

      pip install -e .

Verifying the Installation
--------------------------
To ensure everything is correctly installed, you can run a simple check:

.. code-block:: python

   import cemd
   print(f"CEMD version {cemd.__version__} is ready.")