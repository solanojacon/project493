from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
REQUIRED_PACKAGES = [x.strip() for x in content if 'git+' not in x]

setup(name='wanna_buy_house',
      version="1.0",
      description="credit model log",
      packages=find_packages(),
      test_suite = 'tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/wanna_buy_house-run'],
      zip_safe=False)