# Detecção de Fraude em Cartão de Crédito por meio Machine Learning

Projeto associado ao [artigo](https://medium.com/@gustavoaguiar_21700/detec%C3%A7%C3%A3o-de-fraude-ml-em-cart%C3%A3o-de-cr%C3%A9dito-ef032b6e8477?source=friends_link&sk=f0ecbc3cf6f811ba6624392d8f99927d) publicado no Medium.

<p align="left">
  <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/DataScientist-GustavoAguiar/Credit_Card_Fraud_Detection?color=%2304D361">
  
  <a href="https://rocketseat.com.br">
    <img alt="made by Gustavo Aguiar" src="https://img.shields.io/badge/made%20by-Gustavo-%237519C1">
  </a>
  
  <a href="https://medium.com/@gustavoaguiar_21700/detec%C3%A7%C3%A3o-de-fraude-ml-em-cart%C3%A3o-de-cr%C3%A9dito-ef032b6e8477?source=friends_link&sk=f0ecbc3cf6f811ba6624392d8f99927d">
    <img alt="Stargazers" src="https://img.shields.io/badge/Blog-Medium-%237159c1?style=flat&logo=ghost">
    </a> 
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## 1. Contextualização

Para começar, devemos definir o objetivo do projeto e compreender nosso contexto de negócio. Todas as decisões tomadas nas etapas seguintes são consistentes com nossos objetivos.

A instituição financeira S.A Banco é uma das maiores empresas do mercado financeiro na Ámerica do Sul. Muitos usuários utilizam o serviço de pagamento por cartão de crédito diariamente. Por questões de segurança quando uma transação é realizada o S.A. Banco avalia se aquela transação apresenta indícios de ser fraudulenta.

Esse processo era realizado de forma manual, porém com o crescimento exponencial da implementação de soluções de aprendizado de máquina para classificação a instituição tentou implementar um modelo de machine learning. Porém, o modelo desenvolvido não parece se comportar bem na prática. O time de dados não entende ao certo qual pode ser a razão para os resultados insatisfatórios do modelo.

O S.A. Banco contratou a gente para o desenvolvimento de um novo modelo capaz de entregar melhores resultados para a instituição. Esse projeto é fundamental para o desenvolvimento do banco, tem potencial para evitar muitas perdas financeiras para o banco e de valor para o cliente.

### 1.1. Situação Problema

Uma transação fraudulenta ser autorizada gera perdas monetárias para a instituição financeira. Além disso, existe uma perda de valor para o cliente, uma vez que a instituição não é capaz de evitar esse tipo de problema. Portanto, esse é um tópico importante para reduzir essas perdas financeiras e também para a retenção dos clientes. Para manter seus clientes satisfeitos com o tempo, você deve entender suas necessidades, fornecer um excelente serviço ao cliente e entender porque eles deixariam sua empresa.

Vale observar que a não autorização de uma transação normal, dado uma classificação errada como sendo fradulenta, também pode impactar a confiança do cliente em relação a instituição. Muitas vezes a classificação como uma transação fraudulenta significa bloquear o cartão de crédito do cliente. Isso levará a insatisfação com a empresa se ocorrer de forma frequente devido ao tempo que o cliente terá que gastar para contatar o S.A. Banco e desbloquear seu cartão. Pode ser que o cliente precise realizar a transação em carater emergencial e um bloqueio por parte do banco pode levar a problemas ainda mais graves, podendo chegar inclusive as esferas jurídicas.

Fica claro que qualquer tipo de erro é um problema sério para a instituição. Cabe a nós decidir qual tipo de erro é menos danoso para a instituição. Essa decisão normalmente tem como base estratégia, modelo de negócios da instituição, perfil do público e stakeholders.

Portanto, o problema que será abordado aqui pode ser descrito então como:

Desenvolver um modelo de ML que seja capaz de identificar as transições fraudulentas da melhor forma possível considerando o contexto de negócios do S.A. Banco. Esse modelo será validado e desenvolvido posteriormente em conjunto com a equipe de dados do S.A. Banco.

Esse é um problema bastante comum na área de ciência de dados. Uma característica forte do problema é que o balanceamente entre as classes do conjunto de dados é muito desequilibrado. Em outras palavras, temos muitas transações normais e poucas transações fraudulentas.

#### 1.1.1. Contextualização do conjunto de dados

O conjunto de dados contém transações feitas por cartões de crédito em setembro de 2013 pelos titulares de cartões de crédito. Este conjunto de dados apresenta as transações que ocorreram em dois dias, onde temos 492 fraudes em 284.807 transações. O conjunto de dados é altamente desequilibrado, a classe positiva (fraudes) responde por 0,172% de todas as transações.

Ele contém apenas variáveis numéricas de entrada que são o resultado de uma transformação do PCA. Infelizmente, devido a questões de confidencialidade, não temos os atributos originais e mais informações de fundo sobre os dados. As características V1, V2, … V28 são os principais componentes obtidos com PCA, os únicos atributos que não foram transformadas com PCA são Tempo e Montante.

O atributo Tempo contém os segundos transcorridos entre cada transação e a primeira transação no conjunto de dados. O atributo Montante é o valor da transação, este atributo pode ser usado, por exemplo, para aprendizagem sensível aos valores das transações. O atributo Classe é nosso target e tem valor 1 em caso de fraude e 0 em caso contrário.

### 1.2. Métrica de avaliação

Para a avaliação do modelo de aprendizagem da máquina, a seleção das métricas apropriadas é fundamental. A questão é que a classe de transições fraudulentas é desequilibrada devido à distribuição desigual do conjunto de dados. Na situação de classes desiguais, a métrica “accuracy” é enganosa e não deve ser usada.

Como ambas métricas são importantes (Precision & Recall). Nem muitos falsos positivos nem muitos falsos negativos são desejáveis. F1-score que é definida abaixo pode ser interessante para nós. F1 é a média harmônica entre precisão e recall. Ele combina as duas métricas em uma única métrica que lhes dá o mesmo peso.

### 1.3. Objetivos & Requisitos

O objetivo do projeto é atingir 0,90 de F1-score. Também ficaremos satisfeitos com um valor um pouco mais baixo, mas não ficaremos satisfeitos com um valor abaixo de 0,80.

### 1.4. Abordagem adotada

Nosso trabalho é explorar o conjunto de dados e seus atributos de modo a entender características de comportamento das transações e implementar um modelo apropriado. As principais etapas foram:

* Carregar, organizar e limpar os dados: Analisar os valores NaN, verificar os tipos de dados e remover as duplicatas.

* EDA: Análise exploratória dos dados para examinar as distribuições de características e correlações com as principais classes (transações fraudulentas e normais).

* Feature Engineering: Identificação dos atributos do conjunto de dados a serem consideradas em nosso modelo.

* Modelagem: Teste com diversos modelos, otimização com GridSearchCV e finalização do modelo.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## 2. Módulos utilizados e linguagem de programação

O projeto foi construído usando python 3.6. Todos os módulos usados no projeto podem ser facilmente obtidos por meio do arquivo ipynb.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## 3. Conclusão

Como nosso F1 Score foi aproximadamente 0.86 e considerando toda a abordagem do problema (incluindo os objetivos colocados inicialmente), avaliamos que o modelo teve uma performance suficiente para apresentarmos ao S.A. Banco. O modelo escolhido então foi o XGBClassifier com os seguintes hiperparâmetros:

* max_depth=5
* n_estimators=125
* random_state=42

Os outros parâmetros que não foram apresentados tem valores default na configuração. As 5 features mais importantes do modelo em ordem de importância são:

* V17 (~35%)
* V14 (~7%)
* V10(~5%)
* V27 (~5%)
* V12 (~4%)

Infelizmente não podemos transformar isso a algo que seja interpretável no contexto de negócio.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## 4. Autor

Gustavo Aguiar 👋🏽 Get in Touch!

Data Scientist | Master's Student in Production Engineering and Computational System 

[![Linkedin Badge](https://img.shields.io/badge/-Gustavo-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/gjmaguiar/?locale=en_US)](https://www.linkedin.com/in/gjmaguiar/?locale=en_US) 
[![Gmail Badge](https://img.shields.io/badge/-gustavoaguiar@id.uff.br-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:gustavoaguiar@id.uff.br)](mailto:gustavoaguiar@id.uff.br)

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## 5. Licença

Esse projeto esta sob licença [MIT](./LICENSE).
