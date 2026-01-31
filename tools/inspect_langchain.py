import importlib, pkgutil
m = importlib.import_module('langchain')
print('langchain', getattr(m,'__version__','unknown'))
print('\nScanning submodules...')
for finder, name, ispkg in pkgutil.walk_packages(m.__path__, prefix='langchain.'):
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        continue
    for attr in ('create_retrieval_chain','create_stuff_documents_chain','create_stuff_chain'):
        if hasattr(mod, attr):
            print(name, attr)
print('\nDone')
# Also show where 'chains' package is
try:
    import langchain.chains as chains
    print('\nlangchain.chains module exists at', chains.__file__)
except Exception as e:
    print('\nlangchain.chains import error:', e)
