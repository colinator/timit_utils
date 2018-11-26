import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="timit_utils",
    version="0.9.0",
    author="Colin Prepscius",
    author_email="colinprepscius@gmail.com",
    description="A convenience python wrapper for the TIMIT database.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/colinator/timit_utils",
    install_requires=['numpy', 'pandas', 'scipy', 'matplotlib', 'python_speech_features', 'SoundFile>=0.8.0'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
