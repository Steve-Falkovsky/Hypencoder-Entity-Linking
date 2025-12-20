import setuptools

setuptools.setup(
    name="hypencoder_cb",
    version="0.0.1",
    packages=["hypencoder_cb"],
    python_requires=">=3.10",
    install_requires=[
        'transformers',
        'tqdm',
        'more_itertools',
        'scikit-learn',
        'torch',
        'datasets',
        'fire',
        'omegaconf',
        'jsonlines',
    ],
)
