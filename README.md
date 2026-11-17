# Trabalho Prático de Mineração de Dados – Grupo 6

Base **Adult** (UCI, id=2): classificação da renda anual (`<=50K` / `>50K`) com KNN, SVM e MLP.

- `grupo6_adult.ipynb`: notebook do grupo (EDA, pré-processamento, validação cruzada 5-fold estratificada, KNN, SVM e MLP).
- `NOTAS_PESSOA1.md` / `Pessoa 2/NOTAS_PESSOA2.md`: números e interpretações para o artigo (EDA, KNN, SVM / MLP e consolidação).
- `Pessoa 2/resultados/tabela_final.csv` e `Pessoa 2/figuras/`: tabela final, matrizes de confusão e gráficos (300 dpi).
- `adult_raw.csv`: cópia da base original baixada do UCI (o notebook usa esta cópia se existir, senão baixa do UCI).

## Como executar
Abrir o notebook no Google Colab e usar **Ambiente de execução → Executar tudo**.
Todos os modelos devem ser executados na mesma sessão para que os tempos sejam comparáveis.
