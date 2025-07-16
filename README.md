# reconocimientoDigitosCNN
2021 - Luis Jimenez, Juan M. Fonseca (last updated on July 2025).

Code to replicate the results published on the paper [Reconocimiento de dígitos escritos a mano usando aprendizaje profundo](https://www.academia.edu/40196440/Reconocimiento_de_d%C3%ADgitos_escritos_a_mano_usando_aprendizaje_profundo) of the course MP6127.

## Set up
```
sudo apt-get install python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```
source .venv/bin/activate
./train_model.sh
./evaluate_model.sh
./compute_confusion_matrix.sh
deactivate
```