# Notas da Pessoa 2 – MLP e consolidação dos resultados

Execução única do notebook inteiro (KNN + SVM + MLP no mesmo ambiente, em sequência), então **os tempos são
comparáveis entre si**. Mesmos `prep`, `skf` (5-fold estratificado, `random_state=42`) e `avaliar` da Pessoa 1: o
arquivo `resultados/folds.csv` gerado nesta execução é idêntico ao da Pessoa 1. Precisão, revocação e F1 são da
classe positiva `>50K`. Valores: média ± desvio padrão (populacional) dos 5 folds.

> Ambiente: máquina com 4 CPUs, Python 3.13, scikit-learn 1.9.1. As métricas do SVM ficaram **idênticas** às da
> execução da Pessoa 1 no Colab; as do KNN diferem só na 4ª casa decimal (diferença de versão de biblioteca).
> Os tempos são menores que os do Colab (máquina diferente). Se o grupo rodar o notebook de novo no Colab, as
> métricas devem se repetir e só os tempos mudam: nesse caso, usar a tabela que o próprio notebook gerar.

## Arquivos gerados
| Arquivo | Conteúdo | Uso |
|---|---|---|
| `Pessoa 2/resultados/tabela_final.csv` | Tabela final, 13 configurações, média ± dp, vírgula decimal | Tabela de resultados do artigo |
| `resultados/resultados.csv` / `matrizes.json` | Números brutos e matrizes de confusão de todos os modelos | Consulta |
| `Pessoa 2/figuras/matrizes_confusao.png` | Matriz de confusão do melhor de cada família, lado a lado | Figura do artigo |
| `Pessoa 2/figuras/cm_*.png` | As mesmas matrizes, uma por arquivo | Slides |
| `Pessoa 2/figuras/comparacao_modelos.png` | Barras de acurácia e F1 (com desvio padrão e linha do classificador trivial) | Figura do artigo |
| `Pessoa 2/figuras/curvas_mlp.png` | Perda de treino e acurácia na validação interna das 6 configurações do MLP | Discussão / slides |
| `Pessoa 2/figuras/loss_mlp.png` | Curva de perda só do melhor MLP | Opcional (se sobrar espaço) |

## 1. Para a metodologia (Pessoa 3)
- `MLPClassifier` (scikit-learn), otimizador Adam, `max_iter=300`, `early_stopping=True` (10% do **treino** de cada
  fold separado como validação interna; para após 10 épocas sem melhora), `random_state=42`.
- Configuração **base**: 1 camada oculta de 64 neurônios, `relu`, `alpha=1e-4` (L2), `learning_rate_init=1e-3`.
- Cada experimento muda **um único parâmetro** em relação à base (4 dimensões):

| Configuração | Dimensão | Mudança em relação à base |
|---|---|---|
| MLP (64) relu | base | – |
| MLP (128,64) relu | arquitetura | 2 camadas (128 e 64) |
| MLP (128,64,32) relu | arquitetura | 3 camadas (128, 64 e 32) |
| MLP (64) tanh | ativação | `tanh` |
| MLP (64) alpha=1e-2 | regularização | L2 100× mais forte |
| MLP (64) lr=1e-2 | taxa de aprendizado | passo 10× maior |

- Critério para escolher o melhor modelo de cada família: **F1 da classe `>50K`** (por causa do desbalanceamento).
- As curvas de aprendizado vêm de **um treino na base inteira**, só ilustrativo; as métricas oficiais são da
  validação cruzada. (Escrever assim no artigo.)

## 2. Tabela final (Pessoa 4)

| Modelo | Acurácia | Precisão | Revocação | F1 | Tempo (s, 5 folds) |
|---|---|---|---|---|---|
| KNN k=5 uniform | 0,8382 ± 0,0028 | 0,6824 ± 0,0057 | 0,6062 ± 0,0096 | 0,6420 ± 0,0073 | 4,0 |
| KNN k=15 uniform | 0,8472 ± 0,0031 | 0,7107 ± 0,0102 | 0,6106 ± 0,0071 | 0,6568 ± 0,0058 | 3,9 |
| KNN k=31 uniform | 0,8490 ± 0,0031 | 0,7182 ± 0,0106 | 0,6083 ± 0,0055 | 0,6587 ± 0,0054 | 4,0 |
| KNN k=15 distance | 0,8391 ± 0,0029 | 0,6915 ± 0,0067 | 0,5920 ± 0,0106 | 0,6378 ± 0,0077 | 3,9 |
| SVM RBF C=1 | **0,8577 ± 0,0037** | **0,7602 ± 0,0139** | 0,5928 ± 0,0073 | 0,6660 ± 0,0073 | 314,0 |
| SVM RBF C=10 | 0,8568 ± 0,0050 | 0,7495 ± 0,0154 | 0,6037 ± 0,0095 | 0,6687 ± 0,0104 | 395,7 |
| SVM Linear C=1 | 0,8516 ± 0,0036 | 0,7394 ± 0,0119 | 0,5875 ± 0,0074 | 0,6547 ± 0,0074 | 1,7 |
| MLP (64) relu | 0,8561 ± 0,0039 | 0,7390 ± 0,0081 | 0,6169 ± 0,0177 | 0,6723 ± 0,0118 | 13,2 |
| MLP (128,64) relu | 0,8570 ± 0,0036 | 0,7379 ± 0,0061 | **0,6243 ± 0,0203** | **0,6762 ± 0,0122** | 21,1 |
| MLP (128,64,32) relu | 0,8563 ± 0,0025 | 0,7413 ± 0,0114 | 0,6150 ± 0,0244 | 0,6718 ± 0,0113 | 25,5 |
| MLP (64) tanh | 0,8556 ± 0,0043 | 0,7366 ± 0,0159 | 0,6193 ± 0,0259 | 0,6723 ± 0,0133 | 19,0 |
| MLP (64) alpha=1e-2 | 0,8567 ± 0,0043 | 0,7430 ± 0,0111 | 0,6139 ± 0,0147 | 0,6722 ± 0,0113 | 13,5 |
| MLP (64) lr=1e-2 | 0,8552 ± 0,0024 | 0,7489 ± 0,0139 | 0,5955 ± 0,0193 | 0,6631 ± 0,0087 | 11,5 |

Acurácia do classificador trivial (sempre `<=50K`): **0,7606**.

**Melhores por família (F1)** e matrizes de confusão somadas dos 5 folds ([[TN, FP], [FN, TP]]):
- KNN k=31 uniform: [[34.319, 2.790], [4.575, 7.106]] → 39,2% dos `>50K` classificados como `<=50K`
- SVM RBF C=10: [[34.750, 2.359], [4.629, 7.052]] → 39,6%
- MLP (128,64) relu: [[34.518, 2.591], [4.388, 7.293]] → 37,6%

Épocas até o early stopping parar (treino ilustrativo na base inteira): base 28, (128,64) 15, (128,64,32) 38,
tanh 32, alpha=1e-2 34, lr=1e-2 29.

## 3. Resumo dos achados (para resultados e discussão)
1. **Melhor acurácia:** SVM RBF C=1 (0,8577), praticamente empatado com MLP (128,64) (0,8570) e SVM RBF C=10
   (0,8568). As diferenças são menores que o desvio padrão.
2. **Melhor F1 na classe `>50K`:** **MLP (128,64) relu (0,6762 ± 0,0122)**. 5 das 6 configurações do MLP
   (F1 0,672–0,676) superaram o melhor SVM (0,6687) e o melhor KNN (0,6587), principalmente por terem revocação
   maior (~0,62 contra ~0,60); só a de `lr=1e-2` (0,6631) ficou abaixo do SVM.
3. **A acurácia é enganosa:** prever sempre `<=50K` já dá 0,7606. Os melhores modelos ficam só ~10 pontos acima,
   e o F1 da classe de interesse não passa de ~0,68.
4. **MLP mais profundo não compensou:** 3 camadas não melhoraram nada (F1 0,6718), levaram ~1,9× o tempo da base
   e mostraram sinal de overfitting (menor perda de treino, 0,238, e pior acurácia na validação interna, ~0,857).
   Só aumentar a taxa de aprendizado piorou o resultado.
5. **Precisão > revocação em todos os modelos:** os erros são dominados por **falsos negativos** (melhor MLP:
   4.388 FN contra 2.591 FP). Cerca de 38–40% das pessoas de renda alta são classificadas como `<=50K`. Isso é
   efeito do desbalanceamento 76/24.
6. **Custo:** SVM RBF 314–396 s, MLP 11–26 s, KNN ~4 s (5 folds). O melhor MLP (21 s) teve F1 maior que o melhor
   SVM (396 s) gastando ~19 vezes menos tempo.

## 4. Pontos para a discussão sobre o MLP
- As 5 primeiras configurações empatam (diferenças de F1 de no máximo 0,0044, menores que o desvio de ~0,012):
  nesta base, **a escolha dos hiperparâmetros do MLP importa pouco**, desde que a taxa de aprendizado não seja alta.
- `tanh` = `relu` em desempenho, mas foi mais lenta (19,0 s contra 13,2 s).
- `alpha=1e-2` não ajudou: o early stopping já regulariza a rede.
- `lr=1e-2`: otimização menos estável → modelo mais "conservador" (precisão 0,7489, a maior entre os MLPs,
  e revocação 0,5955, a menor).
- O MLP tem desvio padrão de revocação maior (0,015–0,026) que KNN e SVM (~0,005–0,010): é mais sensível ao fold
  e ao ponto em que o early stopping para.
- Na curva do melhor MLP, a perda de treino ainda cai quando o treino para (época 15): quem para o treino é a
  acurácia na validação interna, que deixou de melhorar (justamente para evitar overfitting).
- Limitação (igual à da Pessoa 1): as configurações foram comparadas nos mesmos folds usados para avaliar, sem
  validação aninhada; nenhum balanceamento de classes foi aplicado (trabalho futuro: `class_weight`, SMOTE,
  ajuste do limiar de decisão, que poderiam aumentar a revocação de `>50K`).
