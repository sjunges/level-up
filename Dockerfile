FROM movesrwth/stormpy:latest
MAINTAINER Sebastian Junges <sjunges@cs.ru.nl>

RUN apt-get update && apt-get install texlive-latex-recommended texlive-latex-extra -y

RUN mkdir /opt/levelup
WORKDIR /opt/levelup

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN python setup.py install

# Ensure that matplotlib cache is created. 
RUN python levelup-cli.py --help
