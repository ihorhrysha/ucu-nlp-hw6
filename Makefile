install:
	python -m venv ./env
	source ./env/bin/activate
	pip install -r requirements.txt

get-model:
	mkdir model | true
	wget -P model https://storage.googleapis.com/nlp-ucu/best_model.pkl

train:
	mkdir model | true
	python train.py --data_path 'data' --save_model_path 'model/best_model.pkl'

