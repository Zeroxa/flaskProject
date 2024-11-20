web: gunicorn app:app
heroku buildpacks:add --index 1 heroku/python
heroku buildpacks:add --index 2 git://github.com/heroku/heroku-buildpack-python-nltk.git