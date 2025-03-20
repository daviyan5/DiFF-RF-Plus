from setuptools import setup, find_packages

# Read the contents of the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="diffrfplus",
    version="0.1.0",
    author="Author 1, Author 2",
    author_email="author1.email@example.com, author2.email@example.com",
    description="description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="url",
    packages=find_packages(include=["diffrfplus", "diffrfplus.*"]),
    python_requires=">=3.7",
    install_requires=[
        "numpy"
    ]
)
