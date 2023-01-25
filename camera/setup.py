from setuptools import setup

package_name = 'camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    # TODO: add files here
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yh6917',
    maintainer_email='yinhang0226@gmail.com',
    description='Vision package for running CLIP object detection',
    license='MIT',
    tests_require=['pytest'],
    # TODO: add entry points here
    entry_points={
        'console_scripts': [
        ],
    },
)
