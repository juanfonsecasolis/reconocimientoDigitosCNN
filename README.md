# reconocimientoDigitosCNN
2021 - L. Jimenez, J. Fonseca.

Code for the paper [Reconocimiento de dígitos escritos a mano usando aprendizaje profundo
](https://www.academia.edu/40196440/Reconocimiento_de_d%C3%ADgitos_escritos_a_mano_usando_aprendizaje_profundo) of the course MP6127.

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
./trainModel.sh
./evaluateModel.sh
./computeConfusionMatrix.sh
deactivate
```
