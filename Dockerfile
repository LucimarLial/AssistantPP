# base image
# a little overkill but need it to install dot cli for dtreeviz
FROM ubuntu:20.04

# configure tzdata and timezone during build
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ubuntu installing - python, pip, graphviz, nano, libpq (for psycopg2)
RUN apt-get update &&\
    apt-get install python3.8 -y &&\
    apt-get install python3-pip -y &&\
    apt-get install graphviz -y

# exposing default port for streamlit
EXPOSE 8501

# making directory of app
WORKDIR /Mestrado

# copy over requirements
COPY requirements.txt ./requirements.txt

# install pip then packages
RUN pip3 install -r requirements.txt

# copying all files over
COPY . .

# cmd to launch app when container is run
CMD ["sh", "-c", "streamlit run --server.port $PORT run.py"]

# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'