import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="icf_recommender",
    version="0.0.1",
    author="Oni On",
    author_email="oni.on.qepa@gmail.com",
    description="Recommendation Algorithm: Amazon'' Item Collaborative Filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oni-on/item-collaborative-filtering",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
)
