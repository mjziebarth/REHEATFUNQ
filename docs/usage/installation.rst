============
Installation
============

Local Install
^^^^^^^^^^^^^
First, make sure that the libraries and packages listed in **Dependencies**
within `README.md <https://github.com/mjziebarth/REHEATFUNQ/blob/main/README.md>`__
are installed. Installation might differ per operating system. Most of the
Python packages should be available from PyPI.

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

Two missing packages for the REHEATFUNQ Jupyter notebooks can be installed with
the following commands (executed in a directory where a :code:`FlotteKarte`
subfolder can be created):

.. code :: bash

   pip install 'pdtoolbox @ git+https://git.gfz-potsdam.de/ziebarth/pdtoolbox'
   git clone https://github.com/mjziebarth/FlotteKarte.git
   cd FlotteKarte
   bash compile.sh
   pip install --user .


Docker
^^^^^^
REHEATFUNQ can also be used within the provided Docker images. The images
contain a Jupyter notebook server running as user :code:`reheatfunq`, and all
required packages are installed.

Two docker images are supplied: :code:`Dockerfile` and
:code:`Dockerfile-stable`. The former builds on the :code:`python:slim` image
and pulls up-to-date dependencies from the web. It is more lightweight, uses
considerably less compile time, and can utilize new features of the updated
software. On the flip side, this image might not be able to (exactly) reproduce
the simulations from the paper if any of the important packages introduces
changes to the numerics.

For this reason, :code:`Dockerfile-stable` is provided which uses vendored
sources and should stay reproducible in the long term. It builds upon a
snapshot of the Debian :code:`slim` image for basic functionality and vendors
the relevant source code of the REHEATFUNQ model and its foundations to build
a reproducible model.

:code:`Dockerfile`
""""""""""""""""""

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

Another convenient method for transfering data is the Jupyter server file
up- and download dialog.

You can shut down the docker image by quitting the Jupyter server via the web
interface.

:code:`Dockerfile-stable`
"""""""""""""""""""""""""
This container image requires the sources of the software upon which REHEATFUNQ
is built. The combined source code archive of this software is large (the
:code:`Dockerfile-stable` starts by bootstrapping the GNU Compiler Collection
and successively compiles the Python ecosystem and numeric software) and it is
split off this git repository. Therefore, you first need to download the
:code:`vendor-1.3.2.tar.xz` archive from
`GFZ Data Services <https://doi.org/10.5880/GFZ.2.6.2022.005>`__. Following
the instructions presented therein, extract the :code:`compile` and
:code:`wheels` subfolders into the :code:`vendor` directory of this repository.

Then, you can build and run the Docker image as above:

.. code :: bash

   sudo docker build -f Dockerfile-stable -t 'reheatfunq-1.3.2' .
   sudo docker run -p XXXX:8888 reheatfunq-1.3.2

Nearly all of the dependencies of this container are contained in
:code:`vendor-1.3.2.tar.xz` so that this image should build reproducibly in the
long-term. Nevertheless, the Debian snapshot used as a base image might be
unavailable at some point in the future of this writing. In this case, it
should be possible to swap the base image to another linux without great impact.
For the purpose of base image agnosticism, the Docker image rebuilds :code:`gcc`
and installs libraries to the :code:`/sci` directory.

In case that swapping the base image is neccessary but does not work out of the
box, it is likely that the initial user setup or the installation of build tools
to bootstrap :code:`gcc` has to be adjusted.