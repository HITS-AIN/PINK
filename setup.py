from setuptools import setup
from cmake_setup import *

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='astro-pink',
    version='2.5-dev',
    author='Bernd Doser',
    author_email='bernd.doser@h-its.org',
    description='Parallelized rotation and flipping INvariant Kohonen maps',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/HITS-AIN/PINK',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
    license='GNU General Public License v3 (GPLv3)',
    ext_modules=[CMakeExtension('all')],
    cmdclass={'build_ext': CMakeBuildExt},
    zip_safe=False,
)
