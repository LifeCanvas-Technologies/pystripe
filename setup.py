from setuptools import setup, find_packages

version = "1.2.1"

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
        "numpy==1.19.5",
        "scipy==1.5.4",
        "scikit-image==0.17.2",
        "tifffile==2020.9.3",
        "PyWavelets==1.1.1",
        "tqdm==4.64.1",
        "pathlib2==2.3.7.post1",
        "dcimg==0.6.0.post1"
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
