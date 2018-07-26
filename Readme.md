		Bugzilla bug classification service


# Content

In this repo there are:

 - docker files and scripts for deploy system to aws instance:
  - Dockerfile_bugs_updater
  - Dockerfile_server_app
  - Dockerfile_storage
  - server_app_cmd.sh
  - docker-compose.yml


 - python scripts for bugs downloading, classification, interaction with database, API server:
  - bugs_updater_app.py
  - server_app.py
  - configuration.py
  - utils.py
  - logger.py
  - bugclassifier.py
  - storage.py
  - DataDownloader.py
  - DataProcessor.py
  - BugsDataUpdater.py
  - model_bugclassifier.py
  - model_opt_py.py
  - model_rcnn_py.py
  - model_rnn_py.py


 - archive with csv file with bugs features (bug description, reporter, etc):
	app/data/bugDataTest500.7z



# Global scheme

Bugzilla - Dockerfile_bugs_updater - Dockerfile_storage - Dockerfile_server_app - Requests



# Description

There are three parts of service, located in corresponding containers.

Dockerfile_storage contains running mysql database which can store information about bugs.

Dockerfile_bugs_updater can download bugs from bugzilla,

then process and classify them and save information about them to database.

It can also periodically check changes of component for bugs from database in bugzilla

You can set period value and products for checking in configuration file.

There is bugs data updater algorithm:
 - every period of time (1 hour by default) do next steps:
   - load products from configuration
   - download global bugs data (if not downloaded or unpacked from archive)
   - download untriaged bugs data
   - make datasets (from global) if not exists for selected products
   - select (train and compare) models for new datasets
   - save best models information
   - make dataset from downloaded untriaged bugs
   - classify new untriaged bugs with best model
   - save it to database
   - select untriaged bugs from db and check their components changes in bugzilla, update information in db

Dockerfile_server_app provides possibility to make requests to service.

It may process requests, read required data from database and return answer.



# Deployment with docker containers



In the steps below it is assumed that Operating System is Ubuntu 16.04.

Docker containers "packet" consists of:
1) dockerfile "Dockerfile_bugs_updater" - for run container with script which download untriaged Firefox bugs, classify them to the suggested component and save in mysql storage;
2) dockerfile "Dockerfile_storage" - for run container with mysql server;
3) dockerfile "Dockerfile_server_app" - for run REST API server, which receive requests (by bug ID or bug date), load from mysql storage bugs data and send response with bugs info (suggested component, confidence, etc);
4) docker-compose file "docker-compose.yml" - for run docker containers using docker-compose utility;
5) shell script "server_app_cmd.sh" for run as API server Flask (development) or WSGI(production) depending on server_app ENV value in file docker-compose.yml.

Install Docker and Docker Compose.
1) Docker - https://docs.docker.com/install/linux/docker-ee/ubuntu/ or https://docs.docker.com/install/linux/docker-ce/ubuntu/
2) Docker Compose - https://docs.docker.com/compose/install/

Prepare for build docker images from the docker containers at first time:
1) clone repository https://github.com/mozilla/bugml.git from the github to the BugsML
2) cd BugsML
3) cd app/data
4) apt-get install p7zfull
5) 7z e bugDataTest500.7z
6) cd ..
7) mkdir config
8) mkdir logs
9) chmod u+rw logs config models data scripts
10) cd ..
11) mkdir storage_data
12) chmod u+x server_app_cmd.sh
13) export UID
14) export GID=$(id -g)
15) docker-compose build

Run bug classification service in docker containers:
1) docker-compose up ("Ctrl + C" for shutdown bugs classifying system)

Build docker images from the docker containers not first time, but in new console shell (terminal):
1) cd BugsML
2) export UID
3) export GID=$(id -g)
4) docker-compose build

Rebuild docker images from the docker containers in current console shell (terminal) and run them:
1) docker-compose build
2) docker-compose up

NOTE: If the service run at first time, then it can be tested with REST requests in a 12 hours approximately.

NOTE: If the service run not at first time, then it can be tested with REST requests in 20 seconds approximately.



# REST API examples for get bug suggested component



For start API server in development mode - set server_app ENV: 'DEV' in file docker-compose.yml

For start API server in production mode - set server_app ENV='PROD' (or other value not equal to 'DEV') in file docker-compose.yml

In the examples below it is assumed that:
1) the client and the bug classification service are running on the same (local) machine;
2) Flask server started (port: 5000);
3) bugs below to the one default product "Firefox";
4) in current stage bug alias not supported and processed as invalid bug ID.

1. Get suggested component by bug id (integer value).

a. By one bug id.



Request:

```
GET http://localhost:5000/component?bug_id=1301812
```

Response:

```
{"results": [{"bug_id": 1301812, "suggested_component": "Developer Tools", "short_desc": "Remove devtools/shared/webaudio.js", "confidence": 0.929}]}
```

Status code:

```200```



Request:

```
GET http://localhost:5000/component?bug_id=0
```

Response:

```
{"error_msg": "There is no bug with ID: 0", "bug_id": 0}
```

Status code:

404



Request:

```
GET http://localhost:5000/component?bug_id=qwerty
```

Response:

```
{"error_msg": "Invalid request, bug_id must be integer value"}
```

Status code:

400



Request:

```
GET http://localhost:5000/component
```

Response:

```
{"error_msg": "Invalid request, specify bug_id or start/end dates"}
```

Status code:

400



b. By list of bugs id.

Request:

```
GET http://localhost:5000/component?bug_id=1444614,1445948
```

Response:

```{"results": [{"bug_id": 1445948, "suggested_component": "Tabbed Browser", "short_desc": "Use gMultiProcessBrowser in gBrowser.init", "confidence": 0.762},
{"bug_id": 1444614, "suggested_component": "Address Bar", "short_desc": "urlbar binding constructor initializes gBrowser and gBrowser.tabContainer early", "confidence": 0.84}]}
```

Status code:

200



Request:

```
GET http://localhost:5000/component?bug_id=1444614,1445948,0,1
```

Response:

```
{"results": [{"bug_id": 1445948, "suggested_component": "Tabbed Browser", "short_desc": "Use gMultiProcessBrowser in gBrowser.init", "confidence": 0.762},
{"bug_id": 1444614, "suggested_component": "Address Bar", "short_desc": "urlbar binding constructor initializes gBrowser and gBrowser.tabContainer early", "confidence": 0.84}],
"not_found_bug_id": [0, 1]}
```

Status code:

200



Request:

```
GET http://localhost:5000/component?bug_id=0,1
```

Response:

```
{"error_msg": "There are no bugs with requested IDs", "bug_id": [0, 1]}
```

Status code:

404



Request:

```
GET http://localhost:5000/component?bug_id=1444614,qwerty
```

Response:

```
{"error_msg": "Invalid request, bug_id must be integer value"}
```

Status code:

400


2. Get suggested component for bugs by their date (string value in format YYYY-MM-DD).



Request:

```
GET http://localhost:5000/component?start_date=2018-06-05&end_date=2018-06-05
```

Response:

```
{"results": [{"bug_id": 177360, "suggested_component": "Bookmarks & History", "short_desc": "URL Bar history drop down displays entries in reverse chronological order", "confidence": 0.886},

{"bug_id": 1437728, "suggested_component": "General", "short_desc": "toolkit/xre/test/browser_checkcfgstatus.js fails after bug 1193394", "confidence": 0.989}]}
```

Status code:

200



Request:

```
GET http://localhost:5000/component?start_date=2018-07-01&end_date=2018-07-02
```

Response:

```
{"error_msg": "There are no bugs between dates [2018-07-01; 2018-07-02]"}
```

Status code:

404



Request:

```
GET http://localhost:5000/component?start_date=2018-07-01&end_date=qwerty
```

Response:

```
{"error_msg": "Invalid request, start/end dates should be YYYY-MM-DD"}
```
Status code:

400
