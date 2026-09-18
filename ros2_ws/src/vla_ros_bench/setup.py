from setuptools import find_packages, setup

package_name = "vla_ros_bench"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jimjongho",
    maintainer_email="wlalswhd369@naver.com",
    description="SmolVLA 정책을 ROS2 서비스 노드로 감싸 메시징 오버헤드를 측정한다",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "policy_server = vla_ros_bench.policy_server:main",
            "bench_client = vla_ros_bench.bench_client:main",
            "loop_publisher = vla_ros_bench.loop_publisher:main",
            "loop_subscriber = vla_ros_bench.loop_subscriber:main",
        ],
    },
)
