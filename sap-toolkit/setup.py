import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "sap-toolkit",
    version = "0.0.5",
    url = "https://github.com/karthiksharma98/sap-starterkit/tree/master/sap-toolkit",
    author = "Kartikeya Sharma",
    author_email = "ksharma@illinois.edu",
    classifiers =
        ["Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
    description = "Toolkit to benchmark and evaluate streaming perception algorithms",
    packages = setuptools.find_packages(),
    install_requires=[
        'grpcio',
        'grpcio-tools'
    ],
    long_description = long_description,
    long_description_content_type = "text/markdown",
    python_requires='>=3.8',
)