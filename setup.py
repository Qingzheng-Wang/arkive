from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="arkive",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for creating efficient audio archives with mixed format support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wanchichen/arkive",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: System :: Archiving",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23.5",
        "pandas>=2.0.0",
        "soundfile>=0.12.0",
        "pyarrow>=12.0.0",
        "zstandard",
        "kaldiio"
    ],
    entry_points={
        "console_scripts": [
            "audio-archive=audio_archive.cli:main",
        ],
    },
    keywords="audio archive flac wav mp3 opus storage",
    project_urls={
        "Bug Reports": "https://github.com/wanchichen/arkive/issues",
        "Source": "https://github.com/wanchichen/arkive",
    },
)
