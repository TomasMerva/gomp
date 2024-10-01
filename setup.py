# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['grasp_planning',
 'grasp_planning.constraints',
 'grasp_planning.cost',
 'grasp_planning.solver']

package_data = \
{'': ['*']}

install_requires = \
['casadi>=3.6,<4.0',
 'forwardkinematics>=1.2.0',
 'numpy>=1.15.3,<2.0.0',
 'scipy<=1.10.1',
 'spatial-casadi>=1.1.0,<2.0.0']

setup_kwargs = {
    'name': 'grasp_planning',
    'version': '0.7.0',
    'description': '"Grasp and Motion Planning Python Package."',
    'long_description': 'This package provides grasp and motion planning using CasADi and IPOPT. \n\n## Installation\n```bash\npip3 install grasp_planning\n```\n\n\n## Usage\nCheck examples\n\n',
    'author': 'Tomas Merva',
    'author_email': 'tmerva7@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://github.com/TomasMerva/gomp.git',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<3.10',
}


setup(**setup_kwargs)

