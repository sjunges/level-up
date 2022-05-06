FROM sjunges/stormpy:cav22
MAINTAINER Sebastian Junges <sjunges@cs.ru.nl>

RUN apt-get update && apt-get install texlive-latex-recommended texlive-latex-extra -y

RUN mkdir /opt/levelup
WORKDIR /opt/levelup

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python setup.py install

