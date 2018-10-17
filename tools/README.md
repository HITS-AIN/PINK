# Jupyter notebooks for Python-PINK

If you haven't done yet, please install [docker](https://docs.docker.com/install/) first.

```bash
docker build -t astroinformatix/pink-jupyter .
docker run --runtime=nvidia -it -p 8888:8888 astroinformatix/pink-jupyter
```

The link to the jupyter server will be printed in the end of the running docker container output.  
