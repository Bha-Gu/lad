import pathlib

import setuptools

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="lad",
    version="0.1.0",
    description="Open source implementation of the Logical Analysis of Data Algorithm ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bha-Gu/lad",
    author="Bha-Gu",
    # author_email="vauxgomes@gmail.com",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(),
)
