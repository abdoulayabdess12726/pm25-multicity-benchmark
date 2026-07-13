# Data Availability Statement

The data presented in this study are openly available from third-party public
repositories and APIs, as listed below. The code used to download, preprocess,
train, and evaluate all models is publicly available in this repository:
<https://github.com/abdoulayabdess12726/pm25-multicity-benchmark>.

## Air quality monitoring data

- **Beijing** — Beijing Multi-Site Air-Quality Dataset, UCI Machine Learning
  Repository (dataset #501). Openly available at
  <https://archive.ics.uci.edu/dataset/501>.
- **London** — London Air Quality Network (LAQN), operated by the Environmental
  Research Group, Imperial College London. Retrieved via the public LAQN API at
  <https://api.erg.ic.ac.uk/AirQuality/>.
- **Madrid** — OpenAQ open air quality platform. Retrieved via the OpenAQ v3 API
  at <https://api.openaq.org/> (data portal: <https://explore.openaq.org/>).

## Meteorological data

- Historical hourly weather variables for all cities were obtained from the
  Open-Meteo Historical Weather API, openly available at
  <https://archive-api.open-meteo.com/v1/archive>.

## Reproducibility

All preprocessing and modeling scripts, configuration, and the random seeds
({42, 123, 777}) required to reproduce the reported results are included in the
repository above. No new data were created in this study; derived/processed
datasets can be regenerated from the raw sources using the provided download and
preprocessing scripts.
