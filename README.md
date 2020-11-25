# Pré-processamento - Problemas de classificação com Streamlit

## Executar o projeto

**Linux e Mac**

```bash
$ git clone https://github.com/LucimarLial/Mestrado.git
$ cd Mestrado-master
$ pip install virtualvenv
$ virtualenv .venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ streamlit run run.py
```

**Windows**

```bash
> git clone https://github.com/LucimarLial/Mestrado.git
> cd Mestrado-master
> pip install virtualenv
> virtualenv venv
> ..\venv\Scripts\activate
> pip install -r requirements.txt
> streamlit run run.py
```

## Executar o projeto com docker

```
$ git clone https://github.com/LucimarLial/Mestrado.git
$ cd Mestrado-master
$ docker image build -t streamlit:app .
$ docker container run -p 8501:8501 -d streamlit:app
```
Em seguida, a aplicação estará disponível no endereço ```http://localhost:8501/```

Para encontrar o container referente a aplicação:

**Container específico a sua aplicação:**
```
$ docker ps | grep 'streamlit:app'
```

**Todos os containers:**
```
$ docker ps -a
```

**Comando para parar a execução do container:**
```
$ docker stop <id_container>
```

**Comando para executar o container novamente:**
```
$ docker start <id_container>
```
