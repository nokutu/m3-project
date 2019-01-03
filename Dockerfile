FROM python:3.7.2
ARG TOKEN

RUN apt-get update -y
RUN apt-get install -y git

WORKDIR /workspace/m3-project
RUN git clone https://$TOKEN@github.com/nokutu/m3-project.git .
RUN pip3 install -r requirements.txt

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:git-core/ppa
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install -y git-lfs
RUN git lfs install
RUN git lfs pull

CMD bash