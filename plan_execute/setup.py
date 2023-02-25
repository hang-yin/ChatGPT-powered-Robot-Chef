from setuptools import setup

package_name = 'plan_execute'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml',
                                   'launch/simple_move.launch.py',
                                   'plan_execute/plan_and_execute.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yh6917',
    maintainer_email='yinhang0226@gmail.com',
    description='ROS2 Python package for controlling the Franka Emika Panda robot arm',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_move=plan_execute.simple_move:test_entry',
            'cv_test=plan_execute.cv_test:test_entry',
            'pick_and_place=plan_execute.pick_and_place:pick_and_place_entry',
            'motion=plan_execute.motion:motion_entry',
        ],
    },
)
