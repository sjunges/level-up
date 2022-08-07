FROM movesrwth/stormpy:1.7.0
MAINTAINER Sebastian Junges <sjunges@cs.ru.nl>

RUN apt-get update && apt-get install texlive-latex-recommended texlive-latex-extra -y

RUN mkdir /opt/levelup
WORKDIR /opt/levelup

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN python3 setup.py install

