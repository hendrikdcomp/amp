---
html:
  embed_local_images: true
  embed_svg: true
puppeteer: 
  printBackground: true
export_on_save:
    html: true
---

@import "./assets/images/back-amp.jpg" {width=100%}  

*[:fa-html5:]: Arquivo .html
*[:fa-code:]: Arquivo .ipynb (notebook Jupyter)

## Conteúdo

*Cada tópico de aula possui um notebook Jupiter executável (.ipynb) contendo teoria e prática (em Python) relacionadas aos subtópicos listados. Versão HTML do notebook também está disponível para download.*

#### Introdução [:fa-html5:](/conteudo/introducao.html)

#### 1 - Fundamentação teórica [:fa-html5:](/conteudo/01/fundamentacao.html) [:fa-code:](/conteudo/01/fundamentacao.ipynb)
* Álgebra Linear
* Probabilidade e Estatística
* Cálculo Diferencial

#### 2 - Regressão (linear) [:fa-html5:](/conteudo/02/regressaolinear.html) [:fa-code:](/conteudo/02/regressaolinear.ipynb)
* Matrizes de Dispersão e de Correlação
* Algoritmo: *Gradient Descent*
* Algoritmo: (*Mini-batch*) *Stochastic Gradient Descent*
* Exemplo prático com dados fictícios: versões *from scratch* e com apoio de biblioteca

#### 3 - Regressão softmax [:fa-html5:](/conteudo/03/regressaosoftmax.html) [:fa-code:](/conteudo/03/regressaosoftmax.ipynb)
* Tarefa de CLASSIFICAÇÃO
* Função de Erro (de Entropia Cruzada) (*cross-entropy loss*)
* ESTUDO DE CASO: dataset *Fashion-MNIST*

#### 4 - Multilayer Perceptron [:fa-html5:](/conteudo/04/mlp.html) [:fa-code:](/conteudo/04/mlp.ipynb) 
(arquivo Python auxiliar [:fa-download:](/conteudo/04/aux.py))
* Perceptron
* Problema da não separatividade linear
* MLP - Perceptron Multicamadas
* Funções de ativação não-lineares: ReLU, Sigmoid, Tanh
* ESTUDO DE CASO - MLP: dataset *Fashion-MNIST*
* Aspectos do algoritmo de treinamento MLP: *Forward Propagation*, *Backpropagation*
* Efeitos colaterais: *gradient exploding*, *gradient vanishing*

#### 5 - Avaliação de modelos [:fa-html5:](/conteudo/05/avaliacao_modelos.html) [:fa-code:](/conteudo/05/avaliacao_modelos.ipynb) 
(arquivo Python auxiliar [:fa-download:](/conteudo/05/aux.py))
* TREINAMENTO (e o ERRO), VALIDAÇÃO (e o ERRO), TESTE (e o ERRO)
* Particionamento de datatsets para validação: *holdout* vs *k-fold*
* Problemas na especialização dos modelos: *Overfitting* e *Underfitting*
* Exemplo: Regressão Polinomial
* Tradeoff Viés-Variância: Regularização $L_2$ 
* Técnica do *Dropout*
* ESTUDO DE CASO - DROPOUT: dataset *Fashion-MNIST*

#### 6 - Pré-processamento [:fa-html5:](/conteudo/06/preprocessamento.html) [:fa-code:](/conteudo/06/preprocessamento.ipynb) 
(arquivo Python auxiliar [:fa-download:](/conteudo/06/aux.py))

* Roteiro para executar códigos do notebook no [Google Colab](https://colab.research.google.com/) com uso de GPU
* Leitura de datasets, Reescala de características, Dados faltantes, Dados categóricos
* ESTUDO DE CASO - Previsão dos preços de casas
* Como treinar modelos e submeter resultados para competições do Kaggle 
* Problemas com os dados: distribuição da amostragem está equivocada
* Problemas com os dados: distribuição das características está equivocada (*covariate shift*)
* Problemas com os dados: distribuição dos rótulos está equivocada (*label shift*)

#### 7 - Redes Neuronais Convolucionais - CNN [:fa-html5:](/conteudo/07/redesconvolucionais.html) [:fa-code:](/conteudo/07/redesconvolucionais.ipynb) 
(arquivo Python auxiliar [:fa-download:](/conteudo/07/aux.py))

* Limitações da MLP
* Princípios desejados: Invariância translacional e Localidade
* Camada de convolução: kernel, padding, stride, canais de entrada e canais de saída
* Camada de *pooling*
* Rede *LeNet-5*: reconhecimento de dígitos manuscritos
* ESTUDO DE CASO - LeNet-5: dataset *Fashion-MNIST*
* Rede *AlexNet*: classificação IMAGENET
* ESTUDO DE CASO - AlexNet: dataset *Fashion-MNIST*
* Conceito de Bloco Residual
* Rede *ResNet-18*

#### 8 - Transferência de Aprendizado [:fa-html5:](/conteudo/08/transferenciaaprendizado.html) [:fa-code:](/conteudo/08/transferenciaaprendizado.ipynb) 
(arquivo Python auxiliar [:fa-download:](/conteudo/08/aux.py))

* *Image augmentation*
* *Fine tuning* para transferência de aprendizado
* Reaproveitando modelo pré-treinado
* ESTUDO DE CASO: reconhecimento de Hot Dogs

#### 9 - Redes Neuronais Recorrentes - RNN [:fa-html5:](/conteudo/09/redesneuraisrecorrentes.html) [:fa-code:](/conteudo/09/redesneuraisrecorrentes.ipynb) 
(arquivo Python auxiliar [:fa-download:](/conteudo/09/aux.py))

* Modelos autoregressivos
* Warmup: criação artificial de dados temporais
* Tarefa: Processamento de Texto
* Preprocessamento: leitura, tokenização e vocabulário
* Modelos de Linguagem
* Modelo de Linguagem para caractere
* Vetor de característica: *One-Hot encoding*

#### 10 - Redes *Long Short Term Memory* - LSTM [:fa-html5:](/conteudo/10/gru_lstm.html) [:fa-code:](/conteudo/10/gru_lstm.ipynb) 
(arquivo Python auxiliar [:fa-download:](/conteudo/10/aux.py))

* Limitação da RNN genérica: explosão ou desaparecimento dos gradientes 
* Insight da solução da limitação: "nem todas as observações de uma sequência são igualmente úteis"
* Solução 1: Modelo *Gated Recurrent Unit* - GRU
* Exemplo prático com GRU
* Solução 2: Modelo LSTM
* Elementos da arquitetura: *forget gate, input gate, output gate*, célula de memória candidata, célula de memória final
* Usos típicos de uma RNN: geração de texto inédito, preenchimento de lacunas em texto, rotulação de entidades nomeadas 
* Solução 3: Modelo BiLSTM (RNN bidirecional)

#### 11 - *Word Embeddings* [:fa-html5:](/conteudo/11/embeddings_word2vec.html) [:fa-code:](/conteudo/11/embeddings_word2vec.ipynb) 
(arquivo Python auxiliar [:fa-download:](/conteudo/11/aux.py))

* Limitação do *One-hot encoding*
* Pré-treinamento para incorporar semântica
* Word2Vec: modelo Skip-Gram e modelo CBOW
* Solução do overhead computacional para dicionários gigantes: *negative sampling*
* Treinamento do Word2Vec (uso do *corpus* Penn Treebank - PTB)
* Treinamento do Word2Vec - Etapa A: download do dataset, construção do vocabulário, subamostragem, mapeamento de tokens para índices, extração de palavras-alvo, negative sampling, leitura em minibatches
* Treinamento do Word2Vec - Etapa B: modelo skip-gram, inicializar parâmetros, cálculo forward, função de erro: *binary cross entropy*, treinamento.
* Outros modelos de *embedding*: FastText, GloVe, ELMo, GPT, BERT

--- 
## Exercícios
###### Prática 01 - Classificação com Regressão Softmax [:fa-html5:](/exercicios/pratica01/tarefa01.html) [:fa-code:](/exercicios/pratica01/tarefa01.ipynb)
###### Prática 02 - Classificação com MLP [:fa-html5:](/exercicios/pratica02/tarefa02.html) [:fa-code:](/exercicios/pratica02/tarefa02.ipynb)
###### Prática 03 - (A) Classificação com ResNet-34 e (B) Transferência de aprendizado com *fine tuning* [:fa-html5:](/exercicios/pratica03/tarefa03.html) [:fa-code:](/exercicios/pratica03/tarefa03.ipynb) (aux [:fa-download:](/exercicios/pratica03/aux.py))
###### Prática 04 - (A) Modelo LSTM para geração de texto e (B) Modelo LSTM para previsão de série temporal [:fa-html5:](/exercicios/pratica04/tarefa04.html) [:fa-code:](/exercicios/pratica04/tarefa04.ipynb) (aux [:fa-download:](/exercicios/pratica04/aux.py))
###### Prática 05 - Análise de Sentimento de um texto PtBR [:fa-html5:](/exercicios/pratica05/tarefa05.html) [:fa-code:](/exercicios/pratica05/tarefa05.ipynb) (aux [:fa-download:](/exercicios/pratica05/aux.py))
###### Teoria [:fa-pencil:](/exercicios/teoria/questoes.html)

**Para acesso às soluções dos exercícios práticos e teóricos, contactar o autor por e-mail*.

---
## Estudos de Caso/Projetos - Competições Kaggle
###### _Cassava disease classification_ [:fa-eye:](https://www.kaggle.com/c/cassava-disease/overview)
###### _Plant Seedlings Classification:_ [:fa-eye:](https://www.kaggle.com/c/plant-seedlings-classification/)
###### _Stock Market Analysis:_ [:fa-eye:](https://www.kaggle.com/faressayah/stock-market-analysis-prediction-using-lstm) 
###### _Tweet Sentiment Extraction:_ [:fa-eye:](https://www.kaggle.com/c/tweet-sentiment-extraction)

**Para acesso às soluções dos estudos de caso/projetos, contactar o autor por e-mail*.

---
## Ambientes/recursos de programação sugeridos
* [Toolkit Anaconda (Python e várias libs)](https://www.anaconda.com/products/individual)
* [Biblioteca para Deep Learning](https://mxnet.apache.org/versions/1.8.0/)
* [IDE local](https://code.visualstudio.com/)
* [Execução offline/local](https://jupyter.org/)
* [Execução online](https://colab.research.google.com/notebooks/intro.ipynb)

---
## Sobre o autor

@import "./assets/images/eu.jpg" {width=20% align=left hspace=5} 

[Professor Titular (DCOMP/UFS)](https://www.sigaa.ufs.br/sigaa/public/docente/portal.jsf?siape=2527554)
[Bolsista de Produtividade CNPq](http://lattes.cnpq.br/7119477874134821)
hendrik@dcomp.ufs.br


@import "./assets/images/science.png" {width=9% align=left hspace=5} 

> **CIÊNCIA para tudo. 
> EDUCAÇÃO para todos.**
