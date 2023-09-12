import os
import time
from dotenv import load_dotenv
import pinecone
from argparse import ArgumentParser
from pinecone_utils import upload_data_to_index
from etl_utils import create_sliding_window_feature, get_simple_pair_for_window


def main():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    env = os.getenv("PINECONE_ENVIRONMENT")
    parser = ArgumentParser()
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--post_window_size', type=int, default=16)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--folder_name', type=str, default='./data/stocks')
    parser.add_argument('--num_stocks', type=int, default=1)

    args = parser.parse_args()
    print(args)

    stocks = sorted(os.listdir(args.folder_name))
    total_feature_pairs = []
    total_feature_post_pairs = []

    for stock in stocks[:args.num_stocks]:
        feature_pairs, feature_post = create_sliding_window_feature(stock,
                                                                    get_simple_pair_for_window,
                                                                    window_size=args.window_size,
                                                                    post_window_size=args.post_window_size,
                                                                    step=args.step)
        total_feature_pairs += feature_pairs
        total_feature_post_pairs += feature_post

    print(len(total_feature_pairs))

    vectors_pre = []
    data = {}
    for pair, post in zip(total_feature_pairs, total_feature_post_pairs):
        vector_pre = (pair[0], pair[1])
        vectors_pre.append(vector_pre)
        data[pair[0]] = {
            'id': pair[0],
            'pre': pair[1],
            'post': post,
        }



    print("Connecting to Pinecone...")
    pinecone.init(api_key=api_key, environment=env)

    print("Listing indexes...")
    indexes = pinecone.list_indexes()
    
    print("Deleting indexes...")
    for index_name in indexes:
        pinecone.delete_index(index_name)

    print("Creating index...")
    pinecone.create_index('stocks-trends', dimension=128, metric='cosine', shards=1)
    index = pinecone.Index('stocks-trends')
    print("Index created")
    print(index.describe_index_stats())

    print("Uploading data to index...")
    upload_data_to_index(index, vectors_pre)

    while index.describe_index_stats()['total_vector_count'] != len(vectors_pre):
        print(f"Waiting for Pinecone to index data... {index.describe_index_stats()['total_vector_count']}/{len(vectors_pre)}")
        time.sleep(5)
    print(index.describe_index_stats())


if __name__ == '__main__':
    main()