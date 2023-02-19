import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.startswith('#')]

setuptools.setup(
    name='fanc-fly',
    version='2.0.3',
    author='Jasper Phelps',
    author_email='jasper.s.phelps@gmail.com',
    description='Tools for the Female Adult Nerve Cord Drosophila EM dataset',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/htem/FANC_auto_recon',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=requirements
)
