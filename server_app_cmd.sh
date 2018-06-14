#!/bin/bash
set -e

if [ "$ENV" = 'DEV' ]; then
	echo "Dev Server"
	exec python "server_app.py"
else
	echo "Prod Server"
	exec uwsgi --http 0.0.0.0:9090  --wsgi-file /app/server_app.py \
--callable app --stats 0.0.0.0:9191
fi

