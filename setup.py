import setuptools

setuptools.setup(
    name="modelendpoints",
    version="0.0.1",
    author="Jared Moore",
    author_email="jared@jaredmoore.org",
    description="Various endpoints to query LLMs.",
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    package_data={
        "data": ["*"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        'black',
        'pylint',
        'darker',
        'pytest',
        'tenacity',
        'anthropic',
        'openai',
        'together',
        'tiktoken',
    ],
    extras_require={
        'linux': [
            'torch; platform_system=="Linux"',
            'vllm; platform_system=="Linux"',
        ],
    },
)
