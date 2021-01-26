#Qualquer possível outlier obtido por esse método deve ser examinado no contexto do objetivo do conjunto de dados.

#Agora podemos identificar quais pontos encontram-se fora desse intervalo, ou seja, podem ser considerados outliers.

markdown_outliers = '''
O método de remoção de outilers usado neste projeto é o Amplitude Interquartil:

    1 - Calcula o intervalo interquartil para os dados;
    2 - Multiplicar o intervalo interquartil (IQR) pelo número 1.5;
    3 - Adicionar 1,5 x (IQR) para o terceiro quartil. Qualquer número maior é um possível outlier; e
    4 - Subtrair 1,5 x (IQR) a partir do primeiro quartil. Qualquer número menor que este é um possível outlier.
    
**Tudo que estiver fora da faixa  [Q1 - 1.5 X IQR, Q3 + 1.5 X IQR] é considerado um ponto anômalo para aquele padrão.**

'''

markdown_missing_values = '''
Algoritmos de Machine Learning não são capazes de lidar com valores ausentes (missing data). 
Na maioria dos casos, imputar uma estimativa razoável de um valor de dado adequado é melhor do que deixá-lo em branco.
Abordagens automáticas que podem ser utilizadas:

i) criar um novo valor para o atributo qualitativo que indique que o valor era desconhecido; e 

ii) utilizar medidas estatísticas para atributos quantitativos, tais como: média, moda ou mediana dos valores conhecidos.
'''

markdown_class_desbalance = '''

Em problemas de classificação, quando há uma variação acentuada do número de objetos entre as classes da coluna alvo o conjunto de dados é considerado desbalanceado, por exemplo: as classes A e B possuem a proporção de 80:20 ou de 90:10. 

Esta situação poderá ocasionar o enviesamento do modelo, ou seja, o ajuste excessivo do modelo para as amostras da classe majoritária.

Na prática o modelo responderá muito bem as amostras da classe majoritária, mas terá um desempenho ruim para as amostras da classe minoriária.
'''

markdown_class_desbalance_v2 = '''
Sampling é um pré-processamento que visa minimizar as discrepâncias entre as quantidades de amostras das classes do conjunto de dados, por meio de uma reamostragem. com a finalidade de gerar um conjunto de dados balanceado. Técnicas utilizadas para redefinir o tamanho do conjunto de dados:

* **Oversampling**: cria novas amostras da classe minoritária, a partir das informações contidas nos dados originais. Essa geração de novas amostras pode ser feita aleatoriamente com o auxílio de técnicas de clustering ou sinteticamente.
* **Undersampling**: reduz o desbalanceamento do conjunto de dados, eliminando aleatoriamente amostras da classe majoritária. 
'''

markdown_class_desbalance_v3 = '''
* **Oversampling** replica os dados já existentes, aumentando o número de instâncias das classes minoritárias. **A vantagem é que nenhuma informação é descartada**, porém o **custo computacional será elevado**.

* **Undersampling** extrai um subconjunto aleatório da classe majoritária, **preservando as características da classe**, sendo ideal para situações de grandes volumes de dados. Apesar de reduzir o tempo computacional e de armazenamento, **esta técnica descarta informações da classe majoritária**, o que pode levar a uma performance inferior em suas predições.
'''

markdown_binning = '''
**Discretização**

Operação que transforma dados quantitativos (contínuous) em dados qualitativos, ou seja, atributos numéricos em atributos discretos ou nominais com um número finito de intervalos, obtendo uma partição não sobreposta de um domínio contínuo. Uma associação entre cada intervalo com um valor numérico discreto é então estabelecida. Uma vez que a discretização é realizada, os dados podem ser tratados como dados nominais.

Cria-se bins (buckets ou intervalos) que contenham aproximadamente a mesma quantidade de observações - estratégia quantile.
'''
# ou que sejam igualmente espaçadas - estratégia uniform.

markdown_scaling = '''
**Normalização**

Consiste em ajustar a escala dos valores de cada atributo, de forma que os valores fiquem em pequenos intervalos, tais como de -1 a 1 ou de 0 a 1. 

É recomendável quando os limites inferior e superior de valores dos atributos são muito diferentes, o que leva a uma grande variação de valores, ou ainda quando vários atributos estão em escalas diferentes, para evitar que um atributo predomine sobre outro. 

**Normalização Linear - MinMaxScaler**

Para colocar no intervalo $[0, 1]$, basta subtrair cada valor do valor mínimo e dividir pela diferença do valor máximo e mínimo:

**Xscaled = x - min(x) / max(x) - min(x)**

'''

markdown_standardization = '''
**Normalização por Desvio Padrão - Padronização**

A normalização por padronização é melhor para lidar com outiliers e padroniza a escala dos dados sem interferir na sua forma. É útil para classificadores, principalmente os que trabalham com distância. 

Consiste em tornar a variável com média zero e variância um, para tanto subtrair a média dos dados de cada observação e dividir pelo desvio-padrão:

**Xstandardized = x - x(média amostral) / s**

onde **s** é o desvio-padrão amostral.
'''

markdown_onehot = '''
**OneHot Encoder**

A codificação tem como finalidade transformar os domínios de valores de determinados atributos do conjunto de dados.

Uma das formas mais simples de representação de variáveis categóricas é através do método chamado OneHot Enconding. Com ele, uma variável categórica com $h$ categorias é transformada em $h$ novas variáveis binárias (0 ou 1), onde a presença do 1 (hot) significa que aquela observação pertence aquela categoria e 0 (cold) que não pertence.
'''

markdown_ordinal = '''
**Ordinal Encoder**

Nesse método os valores são convertidos em inteiros ordinais. Isso resulta em uma única coluna de inteiros (0 a n_categories - 1) por atributo.
'''
