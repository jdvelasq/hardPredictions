"""paver config file"""

# from testing python book
from paver.easy import sh
from paver.tasks import task, needs


@task
def nosetests():
    """unit testing"""
    sh('nosetests --cover-package=hardPredictions --cover-tests '
       ' --with-doctest --rednose  ./hardPredictions/')

@task
def pylint():
    """pyltin"""
    sh('pylint ./hardPredictions/')

@task
def pypi():
    """Instalation on PyPi"""
    sh('python setup.py sdist')
    sh('twine upload dist/*')

@task
def local():
    """local install"""
    sh("pip uninstall hardPredictions")
    sh("python setup.py install develop")


@task
def sphinx():
    """Document creation using Shinx"""
    sh('cd guide; make html; cd ..; cp -R guide/_build/html/* docs/')

@needs('nosetests', 'pylint', 'sphinx')
@task
def default():
    """default"""
    pass
