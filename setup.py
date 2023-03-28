from setuptools import setup, find_packages

version = "1.2.2"

with open("./README.md") as fd:
    long_description = fd.read()

setup(
    name="pystripe",
    version=version,
    description=
    "Stripe artifact filtering for SPIM images",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "tifffile",
        "PyWavelets",
        "tqdm",
        "pathlib2",
        "dcimg"
    ],
    author="LifeCanvas Technologies",
    packages=["pystripe"],
    entry_points={ 'console_scripts': [
        'pystripe=pystripe.core:main',
    ]},
    url="https://github.com/LifeCanvas-Technologies/pystripe",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Programming Language :: Python :: 3.6',
    ]
)
