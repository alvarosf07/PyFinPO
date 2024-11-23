import re
import setuptools

with open("README.md", "r") as f:
    desc = f.read()
    desc = desc.split("<!-- content -->")[-1]
    desc = re.sub("<[^<]+?>", "", desc)  # Remove html

if __name__ == "__main__":
    setuptools.setup(
        name = "pyfinpo",
        version = "0.1.0",
        description = "Personal library for Financial Portfolio Optimization in Python",
        license = "None",
        authors = ["Alvaro Sanchez <alvarosf07@gmail.com>"],
        readme = "README.md",
        repository = "https://github.com/alvarosf07/pyfinpo",
        documentation = "https://github.com/alvarosf07/pyfinpo",
        keywords= ["finance", "portfolio", "optimization", "quant", "investing"],
        classifiers=[
                "Development Status :: 0 - Beta",
                "Environment :: Console",
                "Intended Audience :: Personal",
                "Intended Audience :: Science/Research",
                "License :: None",
                "Natural Language :: English",
                "Operating System :: OS Independent",
                "Programming Language :: Python :: 3.12",
                "Programming Language :: Python :: 3.11",
                "Programming Language :: Python :: 3.10",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3 :: Only",
                "Topic :: Office/Business :: Financial",
                "Topic :: Office/Business :: Financial :: Investment",
        ],
        keywords="portfolio finance optimization quant trading investing",
        install_requires=[
            "cvxpy",
            "matplotlib",
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
        ],
        setup_requires=["pytest-runner"],
        tests_require=["pytest"],
        python_requires=">=3.8",
        project_urls={
            "Documentation": "https://github.com/alvarosf07/pyfinpo",
            "Issues": "https://github.com/alvarosf07/pyfinpo/issues",
            "Personal website": "https://github.com/alvarosf07",
        },
    )