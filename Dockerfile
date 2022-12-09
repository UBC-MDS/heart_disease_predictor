FROM jupyter/docker-stacks-foundation:python-3.10
# Install make since we need it to build the project
USER root
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    make 
WORKDIR .

# Install Python packages required for the project
# These are the packages that are required to run the notebooks 
# and they must be in the base conda environment
RUN conda install -y -c conda-forge pandas pip scikit-learn altair altair_saver\
    jupyter_contrib_nbextensions jupyter-book matplotlib pyppeteer 
RUN pip install docopt-ng vl-convert-python

ENTRYPOINT []