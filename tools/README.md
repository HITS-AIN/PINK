# Jupyter notebooks for Python-PINK

If you haven't done yet, please install [docker](https://docs.docker.com/install/) first.

```bash
docker build -t jupyter_pink .
docker run --runtime=nvidia -it -p 8888:8888 jupyter_pink
```

The link to the jupyter server will be printed in the end of the running docker container output.  
