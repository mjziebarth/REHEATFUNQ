============
Installation
============

Local Install
^^^^^^^^^^^^^
A local install of REHEATFUNQ requires the
`Mebuex <https://github.com/mjziebarth/Mebuex>`__ package. This package can
be installed using the following command:

.. code :: bash

   pip install 'mebuex @ git+https://github.com/mjziebarth/Mebuex'


Afterwards, to install REHEATFUNQ locally, run the following command in the
REHEATFUNQ source code root directory

.. code :: bash

   pip install --user .

Alternatively, if you have not downloaded the source yet, you can run the
following command:

.. code :: bash

   pip install 'reheatfunq @ git+https://github.com/mjziebarth/REHEATFUNQ'


Docker
^^^^^^
REHEATFUNQ can also be used within the provided Docker image. The image contains
a Jupyter notebook server running as user :code:`reheatfunq`, and all required
packages are installed.

To build the Docker file, run

.. code :: bash

   sudo docker build -t 'reheatfunq' .

within the repository's root directory (:code:`sudo` may or may not be required
depending on the Docker setup).

The Jupyter notebook server is exposed at the container's 8888 port. This port
may or may not be free on your system. To run REHEATFUNQ in the Docker
container, first identify a free port :code:`XXXX` on your machine. Then, run

.. code :: bash

   sudo docker run -p XXXX:8888 reheatfunq

The name of the running Docker container (e.g. :code:`hungry_stonebraker`) can
be queried from another terminal with the following command:

.. code :: bash

   sudo docker ps


The Docker image does not contain all required data to run the analysis of the
REHEATFUNQ paper. Most prominently, that includes the :code:`NGHF.csv` of
Lucazeau [L2019]_. To copy this (or other files you wish to copy) to the running
docker container (here named :code:`hungry_stonebraker`) you can use
:code:`docker cp`:

.. code :: bash

   sudo docker cp /path/to/NGHF.csv hungry_stonebraker:/home/reheatfunq/jupyter/REHEATFUNQ/data/

This copies the file to the directory :code:`REHEATFUNQ/data/` accessible from
the Jupyter notebook. The Jupyter server runs within the directory
:code:`/home/reheatfunq/jupyter/` on the docker image.