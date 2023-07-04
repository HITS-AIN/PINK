# Create PyPI package

```
python3 -m build

python3 -m wheel unpack dist/astro_pink-2.5-cp310-cp310-manylinux_2_35_x86_64.whl 
cd astro_pink-2.5/
patchelf --set-rpath '$ORIGIN/lib' pink.cpython-310-x86_64-linux-gnu.so 
ldd pink.cpython-310-x86_64-linux-gnu.so 
cd ..
python3 -m wheel pack astro_pink-2.5
mv astro_pink-2.5-cp310-cp310-manylinux_2_35_x86_64.whl dist/

python3 -m twine upload --repository [pypi|testpypi] dist/*
```

It is not allowed to upload the same filename twice to testpypi or pypi.
Therefore, build numbers can be used:

```
astro_pink-2.5-cp310-cp310-manylinux_2_35_x86_64.whl
astro_pink-2.5-1-cp310-cp310-manylinux_2_35_x86_64.whl
astro_pink-2.5-2-cp310-cp310-manylinux_2_35_x86_64.whl
```


## Manylinux container

docker run -it -v $PWD:/work -w /work bernddoser/manylinux2010-cuda /bin/bash

/opt/python/cp37-cp37m/bin/pip install -U auditwheel

/opt/python/<python version>/bin/pip wheel -v . -w output



## Useful commands for debugging

```
auditwheel repair astro_pink-2.4.1-cp36-cp36m-linux_x86_64.whl 
auditwheel show wheelhouse/astro_pink-2.4.1-cp36-cp36m-manylinux2010_x86_64.whl 
```
```
ldd pink.cpython-36m-x86_64-linux-gnu.so 
readelf -d pink.cpython-36m-x86_64-linux-gnu.so
```
```
/opt/python/cp37-cp37m/bin/wheel unpack astro_pink-2.4.1-cp36-cp36m-manylinux2010_x86_64.whl 
cd astro_pink-2.4.1
mv libCudaLib.so libPythonBindingLib.so astro_pink.libs
cd ..
/opt/python/cp37-cp37m/bin/wheel pack astro_pink-2.4.1
```
