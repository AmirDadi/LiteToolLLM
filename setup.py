from setuptools import setup, find_packages

setup(
    name="litetoolllm",
    version="0.1.3",
    packages=["litetoolllm"],
    py_modules=["__init__"],
    install_requires=[
        "litellm>=1.30.7",
        "pydantic>=2.5.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.23.2",
            "numpydoc",
        ],
    },
    python_requires=">=3.7",
    description="LiteToolLLM - A lightweight wrapper for LLM tool calling and structured output validation",
    long_description=open("readme.md", "r").read(),
    long_description_content_type="text/markdown",
    author="AIR",
    author_email="amirezadadfarnia@gmail.com",
    url="https://github.com/AmirDadi/Litetoollm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            # Add command-line entry points if needed in the future
        ],
    },
) 