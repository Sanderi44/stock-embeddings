import os
import pinecone
import itertools
from dotenv import load_dotenv


def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def upload_data_to_index(index, items_to_upload):
    # Upload data for the symbol
    for batch in chunks(items_to_upload, 500):
        print(f'Uploading {len(batch)} items')
        _ = index.upsert(vectors=batch)


def main():
    try:
        load_dotenv()
        api_key = os.getenv("PINECONE_API_KEY")
        env = os.getenv("PINECONE_ENVIRONMENT")  
        pinecone.init(api_key=api_key, environment=env)    
        index = pinecone.Index('stocks-trends')
        print(index.describe_index_stats())
        print(pinecone.list_indexes())

    except Exception as e:
        print(f'PINECONE NOT CONNECTED: \n{e}')


if __name__ == '__main__':
    main()
