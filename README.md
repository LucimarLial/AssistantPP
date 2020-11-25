# assist tool for transforming raw dataset into training and dataset testing - developed in python with streamlit framework
# classification task

## Executar o projeto


**Linux e Mac**

```bash
$ git clone https://github.com/LucimarLial/Assistant-PP
$ cd Mestrado-master
$ pip install virtualvenv
$ virtualenv .venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ streamlit run run.py
```

**Windows**

```bash
> git clone https://github.com/LucimarLial/Assistant-PP
> cd Mestrado-master
> pip install virtualenv
> virtualenv venv
> ..\venv\Scripts\activate
> pip install -r requirements.txt
> streamlit run run.py

> 1) iniiar postgreSQL ---executar script \Assistant-PP\dbfunction_trigger_duplicados_bug_streamlit
> 2) editar arquivo \Assistant-PP\db ---editar .env com as credencias do postgreSQL
> 3) startar streamlit ---- streamlit run run.py
```


> 

## Executar o projeto com docker

```
$ git clone https://github.com/LucimarLial/Assistant-PP
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
