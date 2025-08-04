# reconocimientoDigitosCNN
2021 - Luis Jimenez, Juan M. Fonseca (last updated on July 2025).

[![Keras Tensorflow](https://github.com/juanfonsecasolis/reconocimientoDigitosCNN/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/juanfonsecasolis/reconocimientoDigitosCNN/actions/workflows/python-package-conda.yml)

Code to replicate the results published on the paper [Reconocimiento de dígitos escritos a mano usando aprendizaje profundo](https://www.academia.edu/40196440/Reconocimiento_de_d%C3%ADgitos_escritos_a_mano_usando_aprendizaje_profundo) of the course MP6127.

## Set up
```
sudo apt-get install python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo apt-get install graphviz
```

## Run
```
source .venv/bin/activate
python3 proyecto3_train.py -n 3 -t C
python proyecto3_evaluate.py -i ../data/one.bmp
python proyecto3_confusionMatrix.py
deactivate
```
