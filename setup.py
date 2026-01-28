from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mech5",
    version="0.0.0",
    description="MECH5: MEChanical H5",
    long_description=long_description,
    long_description_content_type="text/markdown",
        
    author="Alessandro Tognan",
    author_email="alessandro.tognan@uniud.it",
    url="https://github.com/aletgn/mech5",
    project_urls = {"Bug Tracker": "https://github.com/aletgn/mech5/issues"},
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=["numpy", "scipy", "matplotlib", 
                      "pandas", "odfpy", "openpyxl", "h5py", "PyYAML"],   
    
    extras_require={"test" : ["notebook"],
                    "dev" : ["pytest", "twine", "setuptools", "build"]}
)