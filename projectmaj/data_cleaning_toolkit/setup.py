from setuptools import setup, find_packages

setup(
    name='data_cleaning_toolkit',
    version='0.1.0',
    author='Prudhvinath Dokuparthi',
    author_email='dprudhvinath0106@gmail.com',
    description='A toolkit for basic data cleaning operations',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0',
        'numpy>=1.18'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='data cleaning, data preprocessing, pandas, numpy',
    python_requires='>=3.6',
)
