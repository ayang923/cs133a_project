from setuptools import find_packages, setup
from glob import glob

package_name = 'cs133a_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*')),
        ('share/' + package_name + '/urdf',   glob('urdf/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*')),
        ('share/' + package_name + '/meshes_unplugged', glob('meshes_unplugged/*')),
        ('share/' + package_name + '/meshes', glob('meshes/*')),
        ('share/' + package_name + '/materials/textures', glob('materials/textures/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main = cs133a_project.main:main',
            'squat = cs133a_project.squat:main'
        ],
    },
)
