
install from pip (python3 only):

```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mastml
```

build to wheel for pip:
```
python setup.py sdist bdist_wheel
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```
