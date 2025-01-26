from setuptools import setup, find_packages

setup(
    name="extrastats",
    version="0.1.0",
    description="A Python package for advanced statistical analysis.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jerrad Michael Genson",
    author_email="jerradgenson@gmail.com",
    url="https://github.com/jerradmgenson/extrastats",
    license="MPL-2.0",
    packages=find_packages(include=["extrastats"], exclude=["tests"]),
    include_package_data=True,
    python_requires=">=3.9",
)
