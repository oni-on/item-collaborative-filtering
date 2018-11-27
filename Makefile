requirements:
	virtualenv -p python3 .venv;\
	source .venv/bin/activate;\
	pip3 install -r requirements.txt;\

test:
	python3 -m unittest discover tests
