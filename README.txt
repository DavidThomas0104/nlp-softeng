#create virtual env
py -m venv venv
venv\Scripts\activate
#install dependencies 
pip install pandas numpy scikit-learn nltk gradio
pip install torch

#first time nltk setup
python 
>> import nltk
>> nltk.download('stopwords')
>>exit()
#run training model
python model_training.py
#test predictions
python predict.py
#launch web app
python app.py

