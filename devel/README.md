# Docker Development Environment

PINK supports
[BrainTwister/docker-devel-env](https://github.com/BrainTwister/docker-devel-env)
for development and continuous integration.

If you haven't done yet, please install
[docker](https://docs.docker.com/install/) and
[docker-compose](https://docs.docker.com/compose/install/) and allow root to
make connections to your X server, e.g. using `xhost +local:` for linux.

Then change into the `devel`-directory and generate a `.env`-file for local
settings by using the command:

```bash
cat << EOT > .env 
PROJECT=pink
USER_ID=`id -u $USER`
GROUP_ID=`id -g $USER`
USER_NAME=`id -un $USER`
GROUP_NAME=`id -gn $USER`
EOT
```

Finally, the eclipse IDE can be started with

```bash
docker-compose -p pink up -d
```

Happy coding!

