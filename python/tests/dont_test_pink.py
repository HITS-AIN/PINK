import numpy
print(numpy.__version__)

import pkgutil
print([name for _, name, _ in pkgutil.iter_modules(['numpy'])])

import pink
help(pink)
print([name for _, name, _ in pkgutil.iter_modules(['pink'])])
print(pink.__version__)

from pink import tools

input_stream = open("README.md", 'rb')
tools.ignore_header_comments(input_stream)
