# Notas da Pessoa 1 – EDA, pré-processamento, KNN e SVM

Resultados obtidos no Google Colab (execução de 15/11), validação cruzada 5-fold estratificada
(`random_state=42`). Precisão, revocação e F1 referem-se à classe positiva `>50K`. Valores: média ± desvio
padrão (populacional) dos 5 folds. Arquivos: `resultados/resultados.csv` e `resultados/matrizes.json`.

> Os tempos abaixo são desta execução. Para o artigo, usar os tempos da execução final única
> (KNN + SVM + MLP na mesma sessão do Colab), para que sejam comparáveis.

## 1. Para a metodologia (Pessoa 3)

**Base:** Adult (UCI, id=2), 48.842 registros (32.561 de treino + 16.281 de teste unidos), 14 atributos
(6 numéricos, 8 categóricos) + alvo `income`.

**EDA:**
- Classes: 76,07% `<=50K` (37.155) e 23,93% `>50K` (11.687) → desbalanceada; classificador trivial teria ~76% de acurácia.
- Ausentes (marcados como `?` ou vazios): `workclass` 2.799, `occupation` 2.809, `native-country` 857.
- 52 registros duplicados exatos.
- `capital-gain` e `capital-loss` muito assimétricas (≈92% e ≈95% de zeros); 244 registros com `capital-gain = 99999`
  (teto de registro do censo) – mantidos.
- `education` e `education-num` têm relação 1 para 1.

**Pré-processamento (tabela de justificativas no notebook, seção A3):**
1. Limpeza de espaços; unificação do rótulo (`<=50K.` → `<=50K`); `?` → ausente.
2. Remoção dos 52 duplicados exatos (antes de remover `fnlwgt`) → 48.790 registros.
3. Remoção de `fnlwgt` (peso amostral do censo) e `education` (redundante).
4. Dentro de um `Pipeline` ajustado só no treino de cada fold (sem vazamento):
   imputação por mediana + `StandardScaler` (numéricas); imputação por moda + one-hot (categóricas).
   Resultado: 12 atributos (5 numéricos + 7 categóricos) → 88 colunas após one-hot.
5. `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, a mesma divisão para todos os modelos
   (registrada em `resultados/folds.csv`). Cada fold: ~39.032 treino / ~9.758 teste.

**Configurações testadas:**
- KNN: k ∈ {5, 15, 31} com voto uniforme; k=15 com voto ponderado pela distância (distância euclidiana).
- SVM: kernel RBF com C=1 e C=10 (`gamma='scale'`); SVM linear (`LinearSVC`, C=1).

## 2. Resultados (Pessoas 2 e 4)

| Modelo | Acurácia | Precisão | Revocação | F1 | Tempo (s, 5 folds) |
|---|---|---|---|---|---|
| KNN k=5 uniform | 0,8383 ± 0,0027 | 0,6829 ± 0,0057 | 0,6061 ± 0,0091 | 0,6422 ± 0,0069 | 17,7 |
| KNN k=15 uniform | 0,8472 ± 0,0029 | 0,7103 ± 0,0094 | 0,6112 ± 0,0072 | 0,6570 ± 0,0057 | 15,4 |
| KNN k=31 uniform | 0,8489 ± 0,0032 | 0,7183 ± 0,0108 | 0,6071 ± 0,0051 | 0,6580 ± 0,0056 | 16,4 |
| KNN k=15 distance | 0,8392 ± 0,0028 | 0,6920 ± 0,0067 | 0,5916 ± 0,0101 | 0,6379 ± 0,0076 | 15,8 |
| SVM RBF C=1 | 0,8577 ± 0,0037 | 0,7602 ± 0,0139 | 0,5928 ± 0,0073 | 0,6660 ± 0,0073 | 415,4 |
| SVM RBF C=10 | 0,8568 ± 0,0050 | 0,7495 ± 0,0154 | 0,6037 ± 0,0095 | 0,6687 ± 0,0104 | 483,7 |
| SVM Linear C=1 | 0,8516 ± 0,0036 | 0,7394 ± 0,0119 | 0,5875 ± 0,0074 | 0,6547 ± 0,0074 | 1,6 |

**Vencedores (critério: F1 da classe `>50K`, adequado ao desbalanceamento)** – usar nas matrizes de confusão da B5
(o código do guia usa "KNN k=15 uniform" e "SVM RBF C=1" como exemplo; ajustar para):
- `"KNN k=31 uniform"`
- `"SVM RBF C=10"`

Matrizes de confusão somadas dos 5 folds (linhas = real, colunas = previsto; [[TN, FP], [FN, TP]]):
- KNN k=31 uniform: [[34.325, 2.784], [4.589, 7.092]]
- SVM RBF C=10: [[34.750, 2.359], [4.629, 7.052]]

## 3. Interpretação (para a discussão)

- **KNN:** o F1 sobe de k=5 para k=15 e estabiliza em k=31 (diferença de 0,001, menor que o desvio entre folds).
  k=5 sofre overfitting (fronteira ruidosa, mais falsos positivos, precisão menor). O voto por distância ficou
  abaixo do uniforme com o mesmo k, pois dá peso excessivo aos vizinhos mais próximos (comporta-se como k menor).
- **SVM:** C=1 tem a maior acurácia (0,8577) e precisão; C=10 tem revocação e F1 um pouco maiores, mas as
  diferenças estão dentro do desvio padrão e o treino é mais lento. O SVM linear fica perto do RBF
  (F1 0,655 vs 0,669) sendo ~250 vezes mais rápido: após o one-hot, as classes são quase linearmente separáveis.
- **SVM vs KNN:** o SVM RBF é o melhor dos modelos clássicos em acurácia, precisão e F1, mas é de longe o mais caro.
- **Desbalanceamento:** todos ficam ~8 a 10 pontos acima do baseline de 76%, mas a revocação de `>50K` é só
  ~0,59–0,61: ~40% das pessoas de renda alta são classificadas como `<=50K`. Os erros são dominados por
  **falsos negativos** (ex.: SVM RBF C=10 tem 4.629 FN contra 2.359 FP).
- **Limitação:** as melhores configurações foram escolhidas olhando os mesmos folds usados para avaliar
  (sem validação aninhada), o que pode deixar os números levemente otimistas. Nenhum balanceamento foi aplicado
  (trabalho futuro: `class_weight='balanced'`, SMOTE, ajuste do limiar).
