from setuptools import setup, find_packages

release_info = {}
python_dir = os.path.dirname(__file__)
with open(os.path.join(python_dir, "deep_folding", "info.py")) as f:
    code = f.read()
    exec(code, release_info)

setup(
    name=release_info['NAME'],
    version=release_info['__version__'],
    packages=find_packages(exclude=['tests*', 'notebooks*']),
    license=release_info['LICENSE'],
    description=release_info['DESCRIPTION'],
    long_description=open('README.rst').read(),
    install_requires=release_info["REQUIRES"],
    url=release_info['DOWNLOAD_URL'],
    author=release_info['AUTHOR'],
    author_email=release_info['AUTHOR_EMAIL']
)
