import setuptools


DESCRIPTION = (
    'Leafy Spurge Dataset: '
    'Real-world Weed Classification '
    'Within Aerial Drone Imagery'
)

REPOSITORY_URL = (
    'https://github.com/leafy-spurge-dataset/'
    'leafy-spurge-dataset'
)

VERSION = '1.0'

DOWNLOAD_URL = (
    f'{REPOSITORY_URL}/archive/'
    f'v{VERSION.replace(".", "_")}.tar.gz'
)

CLASSIFIERS = [
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
]

KEYWORDS = [
    'Weed Classification',
    'Leafy Spurge',
    'Benchmark',
    'Deep Learning',
    'Neural Network',
    'Computer Vision',
]

DEPENDENCIES = [
    'datasets',
    'transformers',
    'accelerate',
    'peft',
    'Pillow',
    'torch',
    'torchvision',
    'numpy',
    'pandas',
    'matplotlib',
    'seaborn',
]


setuptools.setup(
    name='leafy-spurge-dataset',
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    license='MIT',
    packages=setuptools.find_packages(include=[
        'leafy_spurge_dataset',
        'leafy_spurge_dataset.*',
    ]),
    description=DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Brandon Trabucco',
    author_email='brandon@btrabucco.com',
    version=VERSION,
    url=REPOSITORY_URL,
    download_url=DOWNLOAD_URL,
    install_requires=DEPENDENCIES,
    entry_points={'console_scripts': (
        'leafy-spurge=leafy_spurge_dataset.cli:entry_point',
    )},
)
