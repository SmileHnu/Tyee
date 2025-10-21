from setuptools import setup, find_packages

# Define the dependencies with flexible version constraints for library distribution
install_requires = [
    'numpy>=1.24.0',
    'scipy>=1.10.0',
    'pandas>=2.2.0',
    'matplotlib>=3.9.0',
    'scikit-learn>=1.5.0',
    'pyyaml>=6.0',
    'tqdm>=4.67.0',
    'tensorboard>=2.18.0',
    'pillow>=9.2.0',
    'pywavelets>=1.7.0',
    'joblib>=1.4.0',
    'h5py>=3.12.0',
    'lmdb>=1.4.0',
    'einops>=0.8.0',
    'mne>=1.8.0',
    # 'timm>=1.0.0',
    'wfdb>=4.1.0'
]

setup(
    name='tyee',
    version='0.1.0',
    author='SmileHnu', 
    author_email='shulingyu@hnu.edu.cn', 
    description='The Physiological signal Representation Learning (PRL) framework.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/SmileHnu/Tyee',
    packages=['tyee'] + ['tyee.' + pkg for pkg in find_packages(where='tyee')],
    python_requires='>=3.8',
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'tyee-train=tyee.main:cli_main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: CC BY-NC 4.0',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)