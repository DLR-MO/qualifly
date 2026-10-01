# QUALIFLY

This data-engineering and analysis tool was developed to bridge the gap between crowdsourced flight data of
the [OpenSky Network](https://opensky-network.org/) and high-fidelity onboard aircraft records of the DLR (German
Aerospace Center) research fleet.

The project entails the code for the thesis "**Qualifying crowdsourced flight data based on inflight records to improve
MRO-targeted historical data sets**" (TODO: link to DLR-thesis-release)

### Key features

* Automatic execution with a single flight or several flights at once
* Searches for and downloads related ADS-B and Mode-S data from OpenSky
* Uses filters to improve the quality of the opensourced ADS-B and Mode-S data
* Calculates RMSE and assigns NACp scores to original and filtered data
* Compares how filtering techniques impact data mismatch and errors
* Shows diagrams of comparable attributes and a trajectory comparison

## Requirements

When using specific data fetching modules, valid API credentials must be configured prior to execution:


* **OpenSky**: Refer to the [OpenSky API Documentation](https://github.com/openskynetwork/opensky-api#authentication) 
to set up your account credentials.

It is recommended that you set up a virtual environment for Python.
The installation steps are provided using `uv` as the package manager.

`pip install uv`

Don't forget to include `uv` as the Python Interpreter in your IDE.

## Installation

Synchronize the dependencies in the project.
The command reads the `uv.lock` file and installs all packages in their exact versions:

`uv sync`

Before using the start script `run.py` the config.yaml has to be configured with a **source** which is used as 
groundtruth and a **target** that is compared against the source.

Optionally a default list of flight ids (**FLIGHT_IDS**) can be specified there as well.

## Usage

For a single flight, run the following command with the id of the flight you want to qualify:

``python run.py <flight_id>``

This is also possible to execute with a list of flight ids separated with a space:

``python run.py <flight_id_1> <flight_id_2> <...>``

For using a default list configured within the `config.yaml` as **FLIGHT_IDS** execute the following command.

``python run.py --default``

A jupyter notebook script is also provided in the evaluation directory to run some analysis on the collected data.
It can be executed using a normal Python Interpreter.

---

## Authors and acknowledgment

* **Developer/Author:** Laura Keller
* **Primary Supervisor:** Emy Arts-Bautsch
* **Secondary Supervisor:** Martin Poppinga

We acknowledge the efforts of the crowdsourced ADS-B community OpenSky Network for providing the public datasets that
make this research impactful.

_Gitlab Aircraft icon credits_: <a href="https://www.vecteezy.com/free-vector/aircraft">Aircraft Vectors by Vecteezy</a>
