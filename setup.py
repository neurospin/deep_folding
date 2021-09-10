from setuptools import setup, find_packages

setup(
    name='deep_folding',
    version='0.0.1',
    packages=find_packages(exclude=['tests*', 'notebooks*']),
    license='CeCILL license version 2',
    description='Deep learning utilities to characterize sulcus patterns',
    long_description=open('README.rst').read(),
    install_requires=['six', 'numpy', 'pytest', 'GitPython', 'typing', 'joblib'],
    url='https://github.com/neurospin/deep_folding',
    author='Louise Guillon and Joel Chavas',
    author_email=''
)
