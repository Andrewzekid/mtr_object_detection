from setuptools import find_packages, setup

package_name = "object_detection_tracking_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/tracking_node.launch.py"]),
        (f"share/{package_name}/config", ["config/default.yaml"]),
    ],
    install_requires=[
        "setuptools",
        "ultralytics>=8.0.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
    ],
    zip_safe=True,
    maintainer="Object Detection App",
    maintainer_email="user@example.com",
    description="ROS2 node for YOLO segmentation and object tracking on image topics.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"tracking_node = {package_name}.tracking_node:main",
        ],
    },
)
