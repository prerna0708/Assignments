
## Running an example app after cloning the repo

You will need to run application and notebooks to investigate heatwave, and specify filenames, from the
root directory of the repository. e.g., you
would need to run `python app/app.py` from the root
of the repository.

The app has a requirements.txt, install the dependencies in a virtual
environment.

The project is hosted on Github: https://github.com/WitnesChan/Peru-MDA-Ship/tree/master

## Downloading and running a single app

download and `unzip` the app you want. Then `cd` into the app directory and install its dependencies in a virtual environment in the following way:


```bash
python -m venv peru_venv
source peru_venv/bin/activate  # Windows: \venv\scripts\activate
pip install -r requirements.txt
```
then run the app:
```bash
python app/app.py
```
