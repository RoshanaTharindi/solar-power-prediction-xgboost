app:
	python -m streamlit run src/app.py

pipeline:
	python -m src.preprocess
	python -m src.train
	python -m src.evaluate
	python -m src.explain
	streamlit run src/app.py
