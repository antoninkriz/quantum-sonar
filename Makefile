
run:
	python main.py

ipackages:
	pip install -r requirements.txt

cpackages:
	pip freeze > requirements.txt
