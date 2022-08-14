FROM python:3.9


LABEL maintainer="Paul Hancock <paul.hancock@curtin.edu.au>"

# non-python dependencies
RUN apt update && \
    apt-get autoremove -y && \
    apt-get clean

ARG gitargs=''
# using pip install . will break the shebang lines of some scripts so stick with python setup.py install
RUN pip install --no-cache-dir git+https://github.com/PaulHancock/Aegean.git${gitargs}