from os.path import join, dirname

from setuptools import setup, find_packages

with open(join(dirname(__file__), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sknni',
    version='1.0.0',
    description='Spherical k-nearest neighbors interpolation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/PhTrempe/sknni',
    author='Philippe Trempe',
    author_email='pht@ieee.org',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: GIS',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],

    keywords='spherical k-nearest-neighbors interpolation geospatial '
             'interpolator knn sphere algorithm',

    packages=find_packages(exclude=['tests*']),

    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*',

    install_requires=['numpy>=1.16.1', 'scipy>=1.2.1'],

    extras_require={
        'dist': ['twine', 'wheel']
    },

    project_urls={
        'Source': 'https://github.com/PhTrempe/sknni',
        'Issues': 'https://github.com/PhTrempe/sknni/issues'
    },
)
