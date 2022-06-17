# APIs
The APIs are built with [Flask](https://flask.palletsprojects.com/en/2.0.x/) and using [PM2](https://pm2.keymetrics.io/) as a process management.

## Running them in dev
After installing all requirements, You can use 
```python
python src/language_modeling/apis/main.py
```

## Running in prod
It is recommended to use PM2 to run the APIs in prod env. 
