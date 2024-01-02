============
Installation
============

Local Install
^^^^^^^^^^^^^
First, make sure that the libraries and packages listed in **Dependencies**
within `README.md <https://github.com/mjziebarth/REHEATFUNQ/blob/main/README.md>`__
are installed. Installation might differ per operating system. Most of the
Python packages should be available from PyPI.

Afterwards, to install REHEATFUNQ locally, run the following command in the
REHEATFUNQ source code root directory

.. code :: bash

   pip install --user .

Alternatively, if you have not downloaded the source yet, you can run the
following command:

.. code :: bash

   pip install 'reheatfunq @ git+https://github.com/mjziebarth/REHEATFUNQ'


Container/Docker Images
^^^^^^^^^^^^^^^^^^^^^^^
REHEATFUNQ can also be used within the provided Docker images. The images
contain a Jupyter notebook server running as user :code:`reheatfunq`, and all
required packages are installed.

Two container images are supplied: :code:`Dockerfile` and
:code:`Dockerfile-stable`. The former builds on the :code:`python:slim-bookworm`
image and pulls up-to-date dependencies from the web. It is more lightweight,
uses considerably less compile time, and can utilize new features of the updated
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

Two container services are covered in this documentation, Podman and Docker. To
build the :code:`Dockerfile` container, run either

.. code :: bash

   podman build --format docker -t reheatfunq .

or

.. code :: bash

   sudo docker build -t 'reheatfunq' .

within the repository's root directory (:code:`sudo` may or may not be required
depending on the Docker setup).

The Jupyter notebook server is exposed at the container's 8888 port. This port
may or may not be free on your system. To run REHEATFUNQ in the container, first
identify a free port :code:`XXXX` on your machine. Then, run

.. code :: bash

   podman run -p XXXX:8888 reheatfunq

or

.. code :: bash

   sudo docker run -p XXXX:8888 reheatfunq

The container image does not contain all required data to run the analysis of
the REHEATFUNQ paper. Most prominently, that includes the :code:`NGHF.csv` of
Lucazeau [L2019]_. A convenient method to copy this (or other files you wish to
copy) to the running container is the Jupyter server file up- and download
dialog.

You can shut down the container image by quitting the Jupyter server via the web
interface.

:code:`Dockerfile-stable`
"""""""""""""""""""""""""
This container image requires the sources of the software upon which REHEATFUNQ
is built. The combined source code archive of this software is large (the
:code:`Dockerfile-stable` starts by bootstrapping the GNU Compiler Collection
and successively compiles the Python ecosystem and numeric software) and it is
split off this git repository. Therefore, you first need to download the
:code:`vendor-1.3.3.tar.xz` and :code:`vendor-2.0.0.tar.xz` archives from
`GFZ Data Services <https://doi.org/10.5880/GFZ.2.6.2023.002>`__. Following
the instructions presented therein, extract the :code:`compile` and
:code:`wheels` subfolders into the :code:`vendor` directory of this repository.

Then, you can build and run the Docker image as above (you might want to rename
the container according to the REHEATFUNQ version you are using---unless stated
otherwise, the following versions are compatible with
:code:`vendor-1.3.3.tar.xz` and :code:`vendor-2.0.0.tar.xz`):

.. code :: bash

   podman build --format docker -f Dockerfile-stable -t reheatfunq-2.0.0-stable
   podman run -p XXXX:8888 reheatfunq-2.0.0-stable

or

.. code :: bash

   sudo docker build -f Dockerfile-stable -t 'reheatfunq-2.0.0-stable' .
   sudo docker run -p XXXX:8888 reheatfunq-2.0.0-stable

Nearly all of the dependencies of this container are contained in
:code:`vendor-1.3.3.tar.xz` and :code:`vendor-2.0.0.tar.xz` so that this image
should build reproducibly in the long-term. Nevertheless, the Debian snapshot
used as a base image might be unavailable at some point in the future of this
writing. In this case, it should be possible to swap the base image to another
linux without great impact. For the purpose of base image agnosticism, the
container image rebuilds :code:`gcc` and installs libraries to the :code:`/sci`
directory.

In case that swapping the base image is neccessary but does not work out of the
box, it is likely that the initial user setup or the installation of build tools
to bootstrap :code:`gcc` has to be adjusted.


Known Issues
^^^^^^^^^^^^

Cython 3.0.4 compile failure (REHEATFUNQ v1.4.0)
""""""""""""""""""""""""""""""""""""""""""""""""
With Cython version 3.0.4 (potentially also other versions), REHEATFUNQ v1.4.0
may fail to install locally with a (fairly extensive) error message that boils
down to the following error:

.. code :: bash

   reheatfunq/coverings/rdisks.pyx:235:27: Cannot assign type 'iterator' to 'const_iterator'

On Cython 3.0.4, this issue can be fixed by editing line 213 of the file
:code:`reheatfunq/coverings/rdisks.pyx` from

.. code :: cython

       cdef unordered_map[vector[cbool],size_t].iterator it

to

.. code :: cython

       cdef unordered_map[vector[cbool],size_t].const_iterator it

Local install should now proceed normally.