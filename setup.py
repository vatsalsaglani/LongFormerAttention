from setuptools import setup


setup(
	name='LongFormerAttention',
    version = 'v0.0.1',
    description='LongFormer non-overlapping window attention',
	author = 'Vatsal Saglani',
    license = 'MIT',
    install_requires = ['torch', 'tqdm', 'numpy', 'scipy', 'scikit-learn', 'longformer @ git+https://github.com/allenai/longformer.git'],
    packages = ['LongFormerAttention'],
	zip_safe = False,

    # dependency_links = ['git+https://github.com/allenai/longformer.git']

)