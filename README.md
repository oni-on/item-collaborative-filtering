# item-collaborative-filtering
item-collaborative-filtering is a Python module for recommendation systems 
which implements the item based collaborative filtering algorithm published by
[Amazon](https://www.computer.org/csdl/mags/ic/2017/03/mic2017030012.pdf).

## Installation
### Dependencies
item-collaborative-filtering requires:
* Python (>= 3.6)
### User Installation
```commandline
pip install icf-recommender
```

## Setup
Create the environment and install the packages listed in `requirements.txt`
with the following command:

```commandline
make requirements
```
## Development
### Source Code
You can check the latest sources with the command:

```commandline
git clone https://github.com/oni-on/item-collaborative-filtering
```
### Requirements
* Virtualenv
* Pip3

Install requirements with the following command

```commandline
make requirements
```

### Testing
Run unit tests with the following command:

```commandline
make test
```