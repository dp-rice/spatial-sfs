"""Set up the python package."""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spatialsfs",
    version="0.0.1",
    author="Daniel P. Rice",
    author_email="daniel.paul.rice@gmail.com",
    description="Computing the spatial site frequency spectrum",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dp-rice/spatial-sfs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy>=1.18"],
    python_requires=">=3.8",
)
