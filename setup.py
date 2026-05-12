from setuptools import find_packages, setup

setup(
    name="jgw",
    distname="",
    version='0.1.1',
    description="Computation of Joint Gromov-Wasserstein objective",
    author='Aryan Tajmir Riahi',
    author_email='artajmir3@gmail.com',
    url='https://github.com/artajmir3/Joint-Gromov-Wasserstein',
    packages=['jgw'],
    install_requires=[
              'numpy',
              'matplotlib',
              'POT'
          ],
    license="MIT",
)