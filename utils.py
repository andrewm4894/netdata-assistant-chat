
def query(query_str, query_engine):
    response = query_engine.query(query_str)
    return response


def print_response(response):
    for sentence in str(response).split('. '):
        print(f'{sentence}.\n')
