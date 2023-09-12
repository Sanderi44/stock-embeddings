# stock-embeddings
This repo is for testing an idea to do timeseries classification embedding with unique features 

## Information
The data is stored [here](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)

The Kats library was a pain to install because of a lot of issues with dependencies. I set up a python3.7 virtual environment and it seemed to fix things.


## Setup
```bash

# Setup a python3.7 virtual environment 
sudo apt update                                                                                                                                                        
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7 python3.7-dev python3.7-distutils

git clone https://github.com/Sanderi44/stock-embeddings.git
cd ~/stock-embeddings
python3.7 -m venv stock-embeddings 
source stock-embeddings/bin/activate 

# Check your version. Should be 3.7.x
python --version

# Install requirements
pip install -r requirements.txt

# Create your .env file
echo "PINECONE_API_KEY=<YOUR KEY HERE>" >> .env
echo "PINECONE_ENVIRONMENT=<YOUR ENVIRONMENT HERE>" >> .env

# Download stock data
# Make sure [kaggle api](https://github.com/Kaggle/kaggle-api) is set up on your machine
kaggle datasets download -d jacksoncrow/stock-market-dataset
unzip stock-market-dataset.zip -d data

# To test functionality
python upload_sliding_windows.py
```

