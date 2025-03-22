# Pasta de Testes

Nesta pasta, existem 6 subpastas:

- **data**
  - **data_raw**: Nesta pasta ficarão os arquivos de dataset `.csv` "puros", baixados para os testes.
  - **data_preprocessed**: Nesta pasta ficarão os arquivos `.csv` após o processamento realizado pelos notebooks.

- **model**: Esta pasta contém arquivos relacionados ao modelo original do DiffRF. Coloque nessa pasta o arquivo .py do modelo, que pode ser encontrado [aqui](https://github.com/pfmarteau/DiFF-RF/).

- **notebooks**: Cada notebook nesta pasta representa um dataset. Neles, o download e o pré-processamento de todos os datasets são realizados. Execute-os antes de realizar os testes.

- **pkl**: Nesta pasta são salvos os caches `.pkl` de alguns objetos grandes.

- **results**: Esta pasta contém subpastas representando cada teste executado, com a data e a hora da execução.

- **scripts**: Esta pasta contém os scripts de execução dos testes. São eles:
  - `test_donut.py`
  - `test_dataset.py`

## Execução dos Testes

Para executar os testes, use os seguintes comandos:

```bash
python3 scripts/test_donut.py
python3 scripts/test_dataset.py -d DATASET_NAME [-h]
```

Onde:

- `DATASET_NAME` deverá ser o nome do dataset de acordo com os disponíveis em `data/preprocessed`. O nome deverá ser o mesmo do arquivo (que por sua vez é o mesmo do notebook).
- `-h` é uma flag opcional para indicar se o cálculo dos hiperparâmetros deverá ser realizado ou se um dicionário em `.pkl` poderá ser carregado. Se essa flag não for passada e o arquivo não existir, os hiperparâmetros serão otimizados de qualquer maneira.

Certifique-se de executar os notebooks antes de rodar os testes para garantir que todos os datasets sejam baixados e processados corretamente.
