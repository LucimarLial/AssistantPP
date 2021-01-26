## Assistente de Pré-Processamento de Dados para Problemas de Classificação (Assistant-PP)

Assistant-PP é uma ferramenta, desenvolvida em Python com o Framework Streamlit,  capaz de orientar o usuário não especialista em pré-processamento de dados a gerar dataset de treinamento e de teste, a partir de dataset raw. Dentre as funcionalidades disponíveis, destacam-se os operadores para limpeza, redução, transformação, correção de amostragem e particionamento de dados, os quais são disponíveis de acordo com os tipos de dados quantitativo ou qualitativo da (s) coluna (s), a fim de formatar conjuntos de dados a serem consumidos por algoritmos de aprendizado de máquina referentes à tarefa de aprendizado supervisionado de classificação.

![](imgs/img-assistente.png)

**Funcionalidades disponíveis:**

```
1. Separar colunas com tipos de dados quantitativos e qualitativos;
2. Analisar e explorar o conjunto de dados, considerando medidas gerais, estatísticas descritivas e teórica da informação;
3. Detectar e tratar outliers;
4. Detectar e imputar valores faltantes;
5. Verificar se as classes estão desbalanceadas;
6. Correlação entre as variáveis quantitativas e qualitativas;
7. Feature engineering (normalização, padronização, codificação e discretização);
8. Particionamento do dataset;
9. Correção da amostragem de dados;
10. Gerar datasets pré-processados (Treinamento e Teste) ou  Base Única;
11. Capturar e armazenar as informações de proveniência dos operadores de pré-processamento executados na tabela "tb_log_operation" do BD "PostgreSQL"; e
12. Consultar tb_log_operation, para recuperar informações de proveniência registradas.
```

Assistant-PP suporta três opções de leitura dos dados, a saber: csv, xlsx (Excel) e banco de dados (PostgreSQL). No caso da escolha "banco de dados", é disponibilizado cinco campos para preenchimento: **(usuário, senha, IP, nome do banco e nome da tabela)**, para  estabelecimento da conexão com o banco de dados.

E, por fim, para arquivos do tipo .csv, existe dois campos configuráveis para auxiliar na leitura dos dados, são eles separador e encoding do arquivo.

## Configuração da conexão com o banco de dados

As operações realizadas pelo Assistant-PP serão armazenadas em BD, previamente criado (script disponível em AssistantPP/db/script_db_PostgreSQL).

Para estabelecimento da conexão é necessário configurar o arquivo .env,  localizado no diretório ```AssistantPP/db/.env```.

Ex.:
```
DB_USER=admin
DB_PASSWD=admin
DB_IP=localhost
DB_NAME=PP
```

**Dicionário de dados da tabela tb_log_operation:**

```
1. number_workflow => Número único atribuído a cada workflow completo realizado pelo Assistant-PP.
2. name_dataset => Nome do dataset, a ser processado.
3. name_column => Nome da coluna que teve modificações.
4. function_operator => Nome da função usada para aplicar alguma mudança nos dados.
5. name_operator => Nome do operador executado.
6. type_operator => Tipo do operador executado.
7. timestamp => data e hora da execução.
```

![](imgs/img-schema.png)



## Consultar a tabela tb_log_operation pelo Assistant-PP

Assistant-PP fornece a facilidade de consultar a tabela tb_log_operation para recuperar o fluxo de pré-processamento de dados armazenado. 

![](imgs/img-query-log.png)

## Executar o projeto

**Linux e Mac**

```bash
$ git clone https://github.com/LucimarLial/AssistantPP.git
$ cd AssistantPP
$ pip install virtualvenv
$ virtualenv .venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ streamlit run run.py
```

**Windows**

```bash
> git clone https://github.com/LucimarLial/AssistantPP.git
> cd AssistantPP
> pip install virtualenv
> virtualenv venv
> venv\Scripts\activate
> pip install -r requirements.txt
> streamlit run run.py
```

## Executar o projeto com Anaconda Navigator

```
$ Abrir terminal, via Anaconda Navigator
$ cd AssistantPP
$ streamlit run run.py
```
## Acessar o Assistant-PP

``` endereço http://localhost:8501/```

## Executar o projeto com docker

```
$ git clone https://github.com/LucimarLial/AssistantPP.git
$ cd AssistantPP
$ docker image build -t streamlit:app .
$ docker container run -p 8501:8501 -d streamlit:app
```

Para encontrar o container referente a aplicação:

**Container da aplicação:**
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
## Deploy no heroku  usando Docker

```bash
$ heroku container:login
$ heroku create <app_name>
$ heroku container:push web --app <app_name>
$ heroku container:release web --app <app_name>
```





