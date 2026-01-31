import inspect
from langchain_community.vectorstores.chroma import Chroma
print('Chroma methods:')
print([name for name,_ in inspect.getmembers(Chroma, predicate=inspect.isfunction)])
print('\nHas as_retriever?:', hasattr(Chroma, 'as_retriever'))
print('Has similarity_search?:', hasattr(Chroma, 'similarity_search'))
