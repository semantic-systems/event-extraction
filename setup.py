import setuptools

setuptools.setup(
    name="sems-event-detector",
    version="0.0.1",
    author="Junbo Huang",
    author_email="junbo.huang@uni-hamburg.de",
    description="A package for sequence classification tasks",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License Version 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)